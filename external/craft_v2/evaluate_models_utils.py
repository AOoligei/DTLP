import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import logging
import time
import argparse
import os
import json
from utils.metrics import get_link_prediction_metrics
from models.EdgeBank import edge_bank_link_prediction
from utils.utils import set_random_seed
from utils.utils import NegativeEdgeSampler, NeighborSampler, TIME_SLOT_DICT
from tgb_seq.LinkPred.evaluator import Evaluator 
from utils.DataLoader import Data


PAS_MODELS = ['TPNet', 'CRAFT']


def is_craft_pas_enabled_model(model_name: str, model: nn.Module):
    return model_name == 'CRAFT' and isinstance(model, nn.Sequential)


def compute_tpnet_temporal_embeddings(backbone: nn.Module, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                                      node_interact_times: np.ndarray):
    src_embeddings, dst_embeddings = backbone.compute_src_dst_node_temporal_embeddings(
        src_node_ids=src_node_ids,
        dst_node_ids=dst_node_ids,
        node_interact_times=node_interact_times,
    )
    src_embeddings = torch.nan_to_num(src_embeddings, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
    dst_embeddings = torch.nan_to_num(dst_embeddings, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
    return src_embeddings, dst_embeddings


def compute_tpnet_scores(predictor: nn.Module, src_embeddings: torch.Tensor, dst_embeddings: torch.Tensor,
                         entity_stats: np.ndarray, pair_stats: np.ndarray):
    entity_stats = torch.from_numpy(entity_stats).to(device=src_embeddings.device, dtype=src_embeddings.dtype)
    pair_stats = torch.from_numpy(pair_stats).to(device=src_embeddings.device, dtype=src_embeddings.dtype)
    return predictor(src_emb=src_embeddings, dst_emb=dst_embeddings, entity_stats=entity_stats, pair_stats=pair_stats)


def create_craft_src_history_inputs(neighbor_sampler, src_node_ids: np.ndarray, node_interact_times: np.ndarray,
                                    num_neighbors: int):
    src_neighb_seq, _, src_neighb_interact_times = neighbor_sampler.get_historical_neighbors_left(
        node_ids=src_node_ids,
        node_interact_times=node_interact_times,
        num_neighbors=num_neighbors,
    )
    neighbor_num = (src_neighb_seq != 0).sum(axis=1)

    return {
        'src_neighb_seq': torch.from_numpy(src_neighb_seq),
        'src_neighb_seq_len': torch.from_numpy(neighbor_num),
        'src_neighb_interact_times': torch.from_numpy(src_neighb_interact_times),
        'cur_pred_times': torch.from_numpy(node_interact_times),
    }


def attach_craft_candidate_inputs(neighbor_sampler, shared_inputs: dict, dst_node_ids: np.ndarray,
                                  node_interact_times: np.ndarray):
    test_dst = np.asarray(dst_node_ids, dtype=np.int64).reshape(len(node_interact_times), -1)
    repeated_batch_times = np.broadcast_to(node_interact_times[:, np.newaxis], test_dst.shape)
    dst_last_neighbor, _, dst_last_update_time = neighbor_sampler.get_historical_neighbors_left(
        node_ids=test_dst.flatten(),
        node_interact_times=repeated_batch_times.flatten(),
        num_neighbors=1,
    )
    dst_last_neighbor = np.asarray(dst_last_neighbor).reshape(len(test_dst), -1)
    dst_last_update_time = np.asarray(dst_last_update_time).reshape(len(test_dst), -1)
    dst_last_update_time[dst_last_neighbor == 0] = -100000

    return {
        **shared_inputs,
        'test_dst': torch.from_numpy(test_dst),
        'dst_last_update_times': torch.from_numpy(dst_last_update_time),
    }


def compute_craft_candidate_embeddings(backbone: nn.Module, model_inputs: dict):
    edge_embeddings = backbone.compute_src_dst_edge_embeddings(**model_inputs)
    edge_embeddings = edge_embeddings.reshape(-1, edge_embeddings.shape[-1])
    return torch.nan_to_num(edge_embeddings, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)


def compute_craft_scores(predictor: nn.Module, craft_embeddings: torch.Tensor, entity_stats: np.ndarray,
                         pair_stats: np.ndarray, lambda_pair: float = 1.0):
    craft_embeddings = craft_embeddings.reshape(-1, craft_embeddings.shape[-1])
    entity_stats = torch.from_numpy(entity_stats).to(device=craft_embeddings.device, dtype=craft_embeddings.dtype)
    pair_stats = torch.from_numpy(pair_stats).to(device=craft_embeddings.device, dtype=craft_embeddings.dtype)
    entity_input = torch.cat([craft_embeddings, entity_stats], dim=-1)
    pair_input = torch.cat([craft_embeddings, pair_stats], dim=-1)
    s_entity = predictor.entity_head(entity_input).squeeze(dim=-1)
    s_pair = predictor.pair_head(pair_input).squeeze(dim=-1)
    s_total = s_entity + lambda_pair * s_pair
    return s_entity, s_pair, s_total


def compute_pas_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor, loss_type: str, loss_func: nn.Module,
                     bce_loss_func: nn.Module):
    if loss_type == 'BPR':
        return loss_func(pos_scores, neg_scores)
    if loss_type == 'BCE':
        return (
            bce_loss_func(pos_scores, torch.ones_like(pos_scores)).mean() +
            bce_loss_func(neg_scores, torch.zeros_like(neg_scores)).mean()
        ) / 2.0
    raise NotImplementedError(f'Loss type {loss_type} not implemented for PAS models!')


def format_pas_predicts(scores: torch.Tensor, loss_type: str):
    if loss_type == 'BCE':
        return scores.sigmoid()
    return scores


def flush_logger_handlers(logger: logging.Logger):
    if logger is None:
        return
    for handler in logger.handlers:
        try:
            handler.flush()
        except Exception:
            continue


def maybe_log_eval_progress(logger: logging.Logger, mode: str, batch_idx: int, num_batches: int,
                            stage_start_time: float, last_log_time: float, interval_seconds: float = 300.0):
    if logger is None:
        return last_log_time
    now = time.time()
    should_log = batch_idx == 0 or (batch_idx + 1) == num_batches or (now - last_log_time) >= interval_seconds
    if not should_log:
        return last_log_time
    logger.info(f'{mode} evaluation progress: batch {batch_idx + 1}/{num_batches}, elapsed {now - stage_start_time:.1f}s')
    flush_logger_handlers(logger)
    return now

def evaluate_model_link_prediction_multi_negs(model_name: str, model: nn.Module, neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
                                   evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate_data: Data, loss_func: nn.Module, device: str = 'cpu',
                                   num_neighbors: int = 20, time_gap: int = 2000, mode='test', num_negs=100, loss_type = 'BCE', full_data: Data = None, dataset_name: str = None, collision_check: bool = False, analyze_results: bool = False, stats_bank=None, lambda_pair: float = 1.0, logger: logging.Logger = None):
    """
    evaluate models on the link prediction task
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate negative edge sampler
    :param evaluate_data: Data, data to be evaluated
    :param loss_func: nn.Module, loss function
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :return:
    """
    # Ensures the random sampler uses a fixed seed for evaluation (i.e. we always sample the same negatives for validation / test set)
    assert evaluate_neg_edge_sampler.seed is not None
    evaluate_neg_edge_sampler.reset_random_state()
    craft_pas_enabled = is_craft_pas_enabled_model(model_name=model_name, model=model)
    if model_name not in ['CRAFT']:
        model[0].set_neighbor_sampler(neighbor_sampler)
    if (model_name == 'TPNet' or craft_pas_enabled) and stats_bank is None:
        raise ValueError(f'{model_name} evaluation requires a TemporalStatsBank instance!')
    
    model.eval()

    evaluator=Evaluator()

    if evaluate_data.neg_samples is not None:
        num_negs = evaluate_data.num_neg_samples

    with torch.no_grad():
        # store evaluate losses and metrics
        evaluate_losses, evaluate_metrics = [], []
        eval_start_time = time.time()
        eval_progress_log_time = eval_start_time
        num_eval_batches = len(evaluate_idx_data_loader)
        if logger is not None:
            logger.info(f'{mode} evaluation started with {num_eval_batches} batches (multi_negs={num_negs}).')
            flush_logger_handlers(logger)
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        neg_samples_idx = 0
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices]
            if mode in ['test']:
                to_test_mask=evaluate_data.split[evaluate_data_indices]==2
            else:
                to_test_mask=evaluate_data.split[evaluate_data_indices]==1
            test_neg_sample_idx = np.arange(neg_samples_idx, neg_samples_idx + to_test_mask.sum())
            neg_samples_idx += to_test_mask.sum()
            repeated_batch_src_node_ids = np.repeat(batch_src_node_ids, repeats=num_negs, axis=0)
            original_batch_size = batch_src_node_ids.shape[0]
            real_neg_counts = np.full(original_batch_size, num_negs, dtype=np.int64)
            if evaluate_data.neg_samples is not None:
                # since tgb-seq neg samples are only provided for test sample, 
                test_neg_dst_node_ids = evaluate_data.neg_samples[test_neg_sample_idx]
                batch_neg_dst_node_ids = np.zeros((original_batch_size, num_negs), dtype=np.int32)
                if isinstance(test_neg_dst_node_ids, np.ndarray) and test_neg_dst_node_ids.ndim == 2:
                    batch_neg_dst_node_ids[to_test_mask] = test_neg_dst_node_ids
                else:
                    padded_test_neg_dst_node_ids = np.zeros((to_test_mask.sum(), num_negs), dtype=np.int32)
                    for row_idx, neg_row in enumerate(test_neg_dst_node_ids):
                        neg_row = np.asarray(neg_row).reshape(-1)
                        real_neg_counts[np.where(to_test_mask)[0][row_idx]] = len(neg_row)
                        padded_test_neg_dst_node_ids[row_idx, :len(neg_row)] = neg_row
                    batch_neg_dst_node_ids[to_test_mask] = padded_test_neg_dst_node_ids
                if (~to_test_mask).sum() > 0:
                    if collision_check:
                        not_test_neg_dst_node_ids = evaluate_neg_edge_sampler.sample_with_time_collision_check(num_negs, batch_src_node_ids[~to_test_mask], batch_node_interact_times[~to_test_mask], neighbor_sampler)
                    else:
                        _, not_test_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(num_negs*len(batch_src_node_ids[~to_test_mask]))
                        not_test_neg_dst_node_ids = not_test_neg_dst_node_ids.reshape(-1, num_negs)
                    batch_neg_dst_node_ids[~to_test_mask] = not_test_neg_dst_node_ids
            else:
                if collision_check:
                    batch_neg_dst_node_ids=evaluate_neg_edge_sampler.sample_with_time_collision_check(num_negs, batch_src_node_ids, batch_node_interact_times, neighbor_sampler)
                else:
                    _, batch_neg_dst_node_ids=evaluate_neg_edge_sampler.sample(num_negs*len(batch_src_node_ids))
                    batch_neg_dst_node_ids=batch_neg_dst_node_ids.reshape(-1, num_negs)
            batch_neg_dst_node_ids=batch_neg_dst_node_ids.flatten()
            repeated_batch_node_interact_times = np.repeat(batch_node_interact_times, repeats=num_negs, axis=0)

            # we need to compute for positive and negative edges respectively, because the new sampling strategy (for evaluation) allows the negative source nodes to be
            # different from the source nodes, this is different from previous works that just replace destination nodes with negative destination nodes
            if model_name in ['TGAT', 'CAWN', 'TCL']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=repeated_batch_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=repeated_batch_node_interact_times,
                                                                      num_neighbors=num_neighbors)
            elif model_name in ['JODIE', 'DyRep', 'TGN']:
                # note that negative nodes do not change the memories while the positive nodes change the memories,
                # we need to first compute the embeddings of negative nodes for memory-based models
                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_node_interact_times=np.concatenate([batch_node_interact_times, batch_node_interact_times, repeated_batch_node_interact_times], axis=0)
                batch_src_node_embeddings, batch_dst_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                        dst_node_ids=batch_dst_node_ids,
                                                                        neg_dst_node_ids=batch_neg_dst_node_ids,
                                                                        node_interact_times=batch_node_interact_times,
                                                                        edge_ids=batch_edge_ids,
                                                                        edges_are_positive=False,
                                                                        num_neighbors=num_neighbors)
                batch_neg_src_node_embeddings=torch.repeat_interleave(batch_src_node_embeddings, repeats=num_negs, dim=0)
            elif model_name in ['GraphMixer']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                # for i in range(len(repeated_batch_src_node_ids)):
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=repeated_batch_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=repeated_batch_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)
            elif model_name in ['DyGFormer']:
                src_node_ids = np.concatenate([batch_src_node_ids, repeated_batch_src_node_ids], axis=0)
                dst_node_ids = np.concatenate([batch_dst_node_ids, batch_neg_dst_node_ids], axis=0)
                node_interact_times = np.concatenate([batch_node_interact_times, repeated_batch_node_interact_times], axis=0)
                src_node_embeddings, dst_node_embeddings = model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times)
                batch_src_node_embeddings, batch_dst_node_embeddings = src_node_embeddings[:len(batch_src_node_ids)], dst_node_embeddings[:len(batch_dst_node_ids)]
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = src_node_embeddings[len(batch_src_node_ids):], dst_node_embeddings[len(batch_dst_node_ids):]
            elif model_name in ['TPNet']:
                entity_stats_pos, pair_stats_pos = stats_bank.query(src=batch_src_node_ids, dst=batch_dst_node_ids, t=batch_node_interact_times)
                entity_stats_neg, pair_stats_neg = stats_bank.query(
                    src=repeated_batch_src_node_ids,
                    dst=batch_neg_dst_node_ids,
                    t=repeated_batch_node_interact_times,
                )
                batch_src_node_embeddings, batch_dst_node_embeddings = compute_tpnet_temporal_embeddings(
                    backbone=model[0],
                    src_node_ids=batch_src_node_ids,
                    dst_node_ids=batch_dst_node_ids,
                    node_interact_times=batch_node_interact_times,
                )
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = compute_tpnet_temporal_embeddings(
                    backbone=model[0],
                    src_node_ids=repeated_batch_src_node_ids,
                    dst_node_ids=batch_neg_dst_node_ids,
                    node_interact_times=repeated_batch_node_interact_times,
                )
            elif craft_pas_enabled:
                shared_craft_inputs = create_craft_src_history_inputs(
                    neighbor_sampler=neighbor_sampler,
                    src_node_ids=batch_src_node_ids,
                    node_interact_times=batch_node_interact_times,
                    num_neighbors=num_neighbors,
                )
                if shared_craft_inputs is None:
                    continue
                entity_stats_pos, pair_stats_pos = stats_bank.query(src=batch_src_node_ids, dst=batch_dst_node_ids, t=batch_node_interact_times)
                entity_stats_neg, pair_stats_neg = stats_bank.query(
                    src=repeated_batch_src_node_ids,
                    dst=batch_neg_dst_node_ids,
                    t=repeated_batch_node_interact_times,
                )
                batch_pos_inputs = attach_craft_candidate_inputs(
                    neighbor_sampler=neighbor_sampler,
                    shared_inputs=shared_craft_inputs,
                    dst_node_ids=batch_dst_node_ids,
                    node_interact_times=batch_node_interact_times,
                )
                batch_neg_inputs = attach_craft_candidate_inputs(
                    neighbor_sampler=neighbor_sampler,
                    shared_inputs=shared_craft_inputs,
                    dst_node_ids=batch_neg_dst_node_ids.reshape(original_batch_size, -1),
                    node_interact_times=batch_node_interact_times,
                )
                batch_pos_embeddings = compute_craft_candidate_embeddings(
                    backbone=model[0],
                    model_inputs=batch_pos_inputs,
                ).squeeze(dim=1)
                batch_neg_embeddings = compute_craft_candidate_embeddings(
                    backbone=model[0],
                    model_inputs=batch_neg_inputs,
                )
            elif model_name in ['SASRec', 'SGNNHN']:
                neighbor_node_ids, _, _=neighbor_sampler.get_historical_neighbors_left(node_ids=batch_src_node_ids, node_interact_times=batch_node_interact_times, num_neighbors=num_neighbors)
                neighbor_num=(neighbor_node_ids!=0).sum(axis=1)
                batch_neg_dst_node_ids = batch_neg_dst_node_ids.reshape(original_batch_size,-1)
                pos_item = torch.from_numpy(batch_dst_node_ids)
                neg_item = torch.from_numpy(batch_neg_dst_node_ids)
                items = torch.cat([pos_item.unsqueeze(1), neg_item], dim=1)
                batch_data=[torch.from_numpy(neighbor_node_ids), torch.from_numpy(neighbor_num), items]
                positive_probabilities, negative_probabilities = model[0].predict(batch_data)
                negative_probabilities = negative_probabilities.flatten().cpu().numpy()
                positive_probabilities = positive_probabilities.flatten().cpu().numpy()
            else:
                raise ValueError(f"Wrong value for model_name {model_name}!")
            if model_name in ['TPNet']:
                _, _, positive_logits = compute_tpnet_scores(
                    predictor=model[1],
                    src_embeddings=batch_src_node_embeddings,
                    dst_embeddings=batch_dst_node_embeddings,
                    entity_stats=entity_stats_pos,
                    pair_stats=pair_stats_pos,
                )
                _, _, negative_logits = compute_tpnet_scores(
                    predictor=model[1],
                    src_embeddings=batch_neg_src_node_embeddings,
                    dst_embeddings=batch_neg_dst_node_embeddings,
                    entity_stats=entity_stats_neg,
                    pair_stats=pair_stats_neg,
                )
                positive_probabilities = torch.sigmoid(positive_logits.clamp(-30, 30)).cpu().numpy()
                negative_probabilities = torch.sigmoid(negative_logits.clamp(-30, 30)).cpu().numpy()
            elif craft_pas_enabled:
                _, _, positive_scores = compute_craft_scores(
                    predictor=model[1],
                    craft_embeddings=batch_pos_embeddings,
                    entity_stats=entity_stats_pos,
                    pair_stats=pair_stats_pos,
                    lambda_pair=lambda_pair,
                )
                _, _, negative_scores = compute_craft_scores(
                    predictor=model[1],
                    craft_embeddings=batch_neg_embeddings,
                    entity_stats=entity_stats_neg,
                    pair_stats=pair_stats_neg,
                    lambda_pair=lambda_pair,
                )
                positive_probabilities = format_pas_predicts(positive_scores, loss_type).cpu().numpy()
                negative_probabilities = format_pas_predicts(negative_scores, loss_type).cpu().numpy()
            elif model_name not in ['CRAFT', 'SASRec', 'SGNNHN']:
                if 'BCE' in loss_type:
                # get positive and negative probabilities, shape (batch_size, )
                    positive_probabilities = model[1](
                        input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1).sigmoid().cpu().numpy()
                    # get negative probabilities, Tensor, shape (batch_size * num_negative_samples_per_node, )
                    negative_probabilities = model[1](
                        input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1).sigmoid().cpu().numpy()
                elif loss_type == 'BPR':
                    positive_probabilities = model[1](
                        input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1).cpu().numpy()
                    negative_probabilities = model[1](
                        input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1).cpu().numpy()
            batch_neg_dst_node_ids = batch_neg_dst_node_ids.reshape(original_batch_size,-1)
            if model_name == 'TPNet':
                stats_bank.update_batch(src=batch_src_node_ids, dst=batch_dst_node_ids, t=batch_node_interact_times)
                model[0].random_projections.update(
                    src_node_ids=batch_src_node_ids,
                    dst_node_ids=batch_dst_node_ids,
                    node_interact_times=batch_node_interact_times,
                )
            elif craft_pas_enabled:
                stats_bank.update_batch(src=batch_src_node_ids, dst=batch_dst_node_ids, t=batch_node_interact_times)
            if to_test_mask.sum() == 0:
                continue
            positive_probabilities = np.atleast_1d(positive_probabilities)[to_test_mask]
            negative_probabilities = negative_probabilities.reshape(-1,num_negs)[to_test_mask]
            real_neg_counts = real_neg_counts[to_test_mask]
            if len(real_neg_counts) > 0 and np.all(real_neg_counts == real_neg_counts[0]):
                mrr_list = evaluator.eval(y_pred_pos=positive_probabilities, y_pred_neg=negative_probabilities[:, :real_neg_counts[0]])
                evaluate_metrics.extend(mrr_list)
            else:
                for positive_probability, negative_probability_row, real_neg_cnt in zip(positive_probabilities, negative_probabilities, real_neg_counts):
                    mrr_list = evaluator.eval(
                        y_pred_pos=np.asarray([positive_probability]),
                        y_pred_neg=negative_probability_row[:real_neg_cnt][np.newaxis, :],
                    )
                    evaluate_metrics.extend(mrr_list)
            eval_progress_log_time = maybe_log_eval_progress(
                logger=logger,
                mode=mode,
                batch_idx=batch_idx,
                num_batches=num_eval_batches,
                stage_start_time=eval_start_time,
                last_log_time=eval_progress_log_time,
            )
    return {'mrr': np.mean(evaluate_metrics)}


def evaluate_model_link_prediction(model_name: str, model: nn.Module, neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
                                   evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate_data: Data, loss_func: nn.Module, device: str = 'cpu',
                                   num_neighbors: int = 20, time_gap: int = 2000,mode:str='val', loss_type = 'BCE', full_data: Data = None, collision_check: bool = False,dataset_name:str='', stats_bank=None, lambda_pair: float = 1.0, logger: logging.Logger = None):
    """
    evaluate models on the link prediction task
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate negative edge sampler
    :param evaluate_data: Data, data to be evaluated
    :param loss_func: nn.Module, loss function
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :return:
    """
    # Ensures the random sampler uses a fixed seed for evaluation (i.e. we always sample the same negatives for validation / test set)
    assert evaluate_neg_edge_sampler.seed is not None
    evaluate_neg_edge_sampler.reset_random_state()
    craft_pas_enabled = is_craft_pas_enabled_model(model_name=model_name, model=model)
    if model_name not in ['CRAFT']:
        model[0].set_neighbor_sampler(neighbor_sampler)
    if (model_name == 'TPNet' or craft_pas_enabled) and stats_bank is None:
        raise ValueError(f'{model_name} evaluation requires a TemporalStatsBank instance!')

    model.eval()

    with torch.no_grad():
        # store evaluate losses and metrics
        evaluate_losses, evaluate_metrics = [], []
        eval_start_time = time.time()
        eval_progress_log_time = eval_start_time
        num_eval_batches = len(evaluate_idx_data_loader)
        if logger is not None:
            logger.info(f'{mode} evaluation started with {num_eval_batches} batches.')
            flush_logger_handlers(logger)
        evaluate_idx_data_loader_tqdm = tqdm(
            evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices]
            if mode in ['test']:
                to_test_mask=evaluate_data.split[evaluate_data_indices]==2
            else:
                to_test_mask=evaluate_data.split[evaluate_data_indices]==1
            if evaluate_neg_edge_sampler.negative_sample_strategy != 'random':
                batch_neg_src_node_ids, batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(size=len(batch_src_node_ids),
                                                                                                  batch_src_node_ids=batch_src_node_ids,
                                                                                                  batch_dst_node_ids=batch_dst_node_ids,
                                                                                                  current_batch_start_time=batch_node_interact_times[0],
                                                                                                  current_batch_end_time=batch_node_interact_times[-1])
            else:
                if collision_check:
                    batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample_with_time_collision_check(num_negs=1, batch_src_node_ids=batch_src_node_ids, batch_node_interact_times=batch_node_interact_times, neighbor_sampler=neighbor_sampler).flatten()
                    batch_neg_src_node_ids = batch_src_node_ids
                else:
                    batch_neg_src_node_ids, batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(
                        size=len(batch_src_node_ids))

            # we need to compute for positive and negative edges respectively, because the new sampling strategy (for evaluation) allows the negative source nodes to be
            # different from the source nodes, this is different from previous works that just replace destination nodes with negative destination nodes
            if model_name in ['TGAT', 'CAWN', 'TCL']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors)
            elif model_name in ['JODIE', 'DyRep', 'TGN']:
                # note that negative nodes do not change the memories while the positive nodes change the memories,
                # we need to first compute the embeddings of negative nodes for memory-based models
                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_node_interact_times=np.concatenate([batch_node_interact_times,batch_node_interact_times,batch_node_interact_times],axis=0)
                batch_src_node_embeddings, batch_dst_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          neg_dst_node_ids=batch_neg_dst_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          edge_ids=batch_edge_ids,
                                                                          edges_are_positive=False,
                                                                          num_neighbors=num_neighbors)
                batch_neg_src_node_embeddings=batch_src_node_embeddings
            elif model_name in ['GraphMixer']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)
            elif model_name in ['DyGFormer']:
                src_node_ids = np.concatenate([batch_src_node_ids, batch_neg_src_node_ids], axis=0)
                dst_node_ids = np.concatenate([batch_dst_node_ids, batch_neg_dst_node_ids], axis=0)
                node_interact_times = np.concatenate([batch_node_interact_times, batch_node_interact_times], axis=0)
                src_node_embeddings, dst_node_embeddings = model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times)
                batch_src_node_embeddings, batch_neg_src_node_embeddings = src_node_embeddings[:len(batch_src_node_ids)], src_node_embeddings[len(batch_src_node_ids):]
                batch_dst_node_embeddings, batch_neg_dst_node_embeddings = dst_node_embeddings[:len(batch_dst_node_ids)], dst_node_embeddings[len(batch_dst_node_ids):]
            elif model_name in ['TPNet']:
                entity_stats_pos, pair_stats_pos = stats_bank.query(src=batch_src_node_ids, dst=batch_dst_node_ids, t=batch_node_interact_times)
                entity_stats_neg, pair_stats_neg = stats_bank.query(src=batch_neg_src_node_ids, dst=batch_neg_dst_node_ids, t=batch_node_interact_times)
                batch_src_node_embeddings, batch_dst_node_embeddings = compute_tpnet_temporal_embeddings(
                    backbone=model[0],
                    src_node_ids=batch_src_node_ids,
                    dst_node_ids=batch_dst_node_ids,
                    node_interact_times=batch_node_interact_times,
                )
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = compute_tpnet_temporal_embeddings(
                    backbone=model[0],
                    src_node_ids=batch_neg_src_node_ids,
                    dst_node_ids=batch_neg_dst_node_ids,
                    node_interact_times=batch_node_interact_times,
                )
            elif craft_pas_enabled:
                shared_craft_inputs = create_craft_src_history_inputs(
                    neighbor_sampler=neighbor_sampler,
                    src_node_ids=batch_src_node_ids,
                    node_interact_times=batch_node_interact_times,
                    num_neighbors=num_neighbors,
                )
                if shared_craft_inputs is None:
                    continue
                entity_stats_pos, pair_stats_pos = stats_bank.query(src=batch_src_node_ids, dst=batch_dst_node_ids, t=batch_node_interact_times)
                entity_stats_neg, pair_stats_neg = stats_bank.query(src=batch_neg_src_node_ids, dst=batch_neg_dst_node_ids, t=batch_node_interact_times)
                batch_pos_inputs = attach_craft_candidate_inputs(
                    neighbor_sampler=neighbor_sampler,
                    shared_inputs=shared_craft_inputs,
                    dst_node_ids=batch_dst_node_ids,
                    node_interact_times=batch_node_interact_times,
                )
                batch_neg_inputs = attach_craft_candidate_inputs(
                    neighbor_sampler=neighbor_sampler,
                    shared_inputs=shared_craft_inputs,
                    dst_node_ids=batch_neg_dst_node_ids,
                    node_interact_times=batch_node_interact_times,
                )
                batch_pos_embeddings = compute_craft_candidate_embeddings(
                    backbone=model[0],
                    model_inputs=batch_pos_inputs,
                ).squeeze(dim=1)
                batch_neg_embeddings = compute_craft_candidate_embeddings(
                    backbone=model[0],
                    model_inputs=batch_neg_inputs,
                ).squeeze(dim=1)
            elif model_name in ['SASRec', 'SGNNHN']:
                neighbor_node_ids, _, _=neighbor_sampler.get_historical_neighbors_left(node_ids=batch_src_node_ids, node_interact_times=batch_node_interact_times, num_neighbors=num_neighbors)
                neighbor_num=(neighbor_node_ids!=0).sum(axis=1)
                pos_item = torch.from_numpy(batch_dst_node_ids)
                neg_item = torch.from_numpy(batch_neg_dst_node_ids)
                items = torch.cat([pos_item.unsqueeze(1), neg_item.unsqueeze(1)], dim=1)
                batch_data=[torch.from_numpy(neighbor_node_ids), torch.from_numpy(neighbor_num), items]
                positive_probabilities, negative_probabilities = model[0].predict(batch_data)
                negative_probabilities = negative_probabilities.flatten()
            else:
                raise ValueError(f"Wrong value for model_name {model_name}!")
            if model_name in ['TPNet']:
                _, _, positive_logits = compute_tpnet_scores(
                    predictor=model[1],
                    src_embeddings=batch_src_node_embeddings,
                    dst_embeddings=batch_dst_node_embeddings,
                    entity_stats=entity_stats_pos,
                    pair_stats=pair_stats_pos,
                )
                _, _, negative_logits = compute_tpnet_scores(
                    predictor=model[1],
                    src_embeddings=batch_neg_src_node_embeddings,
                    dst_embeddings=batch_neg_dst_node_embeddings,
                    entity_stats=entity_stats_neg,
                    pair_stats=pair_stats_neg,
                )
                positive_probabilities = torch.sigmoid(positive_logits.clamp(-30, 30))
                negative_probabilities = torch.sigmoid(negative_logits.clamp(-30, 30))
            elif craft_pas_enabled:
                _, _, positive_scores = compute_craft_scores(
                    predictor=model[1],
                    craft_embeddings=batch_pos_embeddings,
                    entity_stats=entity_stats_pos,
                    pair_stats=pair_stats_pos,
                    lambda_pair=lambda_pair,
                )
                _, _, negative_scores = compute_craft_scores(
                    predictor=model[1],
                    craft_embeddings=batch_neg_embeddings,
                    entity_stats=entity_stats_neg,
                    pair_stats=pair_stats_neg,
                    lambda_pair=lambda_pair,
                )
                positive_probabilities = format_pas_predicts(positive_scores, loss_type)
                negative_probabilities = format_pas_predicts(negative_scores, loss_type)
            elif model_name not in ['CRAFT', 'SASRec', 'SGNNHN']:
                if 'BCE' in loss_type:
                # get positive and negative probabilities, shape (batch_size, )
                    positive_probabilities = model[1](
                        input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1).sigmoid()
                    negative_probabilities = model[1](
                        input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1).sigmoid()
                elif loss_type == 'BPR':
                    positive_probabilities = model[1](
                        input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1)
                    negative_probabilities = model[1](
                        input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1)
            if model_name == 'TPNet':
                stats_bank.update_batch(src=batch_src_node_ids, dst=batch_dst_node_ids, t=batch_node_interact_times)
                model[0].random_projections.update(
                    src_node_ids=batch_src_node_ids,
                    dst_node_ids=batch_dst_node_ids,
                    node_interact_times=batch_node_interact_times,
                )
            elif craft_pas_enabled:
                stats_bank.update_batch(src=batch_src_node_ids, dst=batch_dst_node_ids, t=batch_node_interact_times)
            if to_test_mask.sum() == 0:
                continue
            positive_probabilities = positive_probabilities[to_test_mask]
            negative_probabilities = negative_probabilities[to_test_mask]
            predicts = torch.cat(
                [positive_probabilities, negative_probabilities], dim=0)
            labels = torch.cat([torch.ones_like(
                positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)
            if loss_func is not None:
                if craft_pas_enabled:
                    if loss_type == 'BPR':
                        evaluate_losses.append(loss_func(positive_scores[to_test_mask], negative_scores[to_test_mask]).item())
                    else:
                        evaluate_losses.append(torch.nn.functional.binary_cross_entropy(predicts, labels).item())
                elif loss_type == 'BPR':
                    evaluate_losses.append(loss_func(positive_probabilities, negative_probabilities).item())
                else:
                        evaluate_losses.append(torch.nn.functional.binary_cross_entropy(predicts, labels).item())
            evaluate_metrics.append(get_link_prediction_metrics(
                predicts=predicts, labels=labels))
            eval_progress_log_time = maybe_log_eval_progress(
                logger=logger,
                mode=mode,
                batch_idx=batch_idx,
                num_batches=num_eval_batches,
                stage_start_time=eval_start_time,
                last_log_time=eval_progress_log_time,
            )
    
    return_metrics={}
    for metric_name in evaluate_metrics[0].keys():
        average_test_metric = np.mean(
            [test_metric[metric_name] for test_metric in evaluate_metrics])
        return_metrics[metric_name] = average_test_metric
    if len(evaluate_losses)>0:
        return_metrics["val_loss"] = np.mean(evaluate_losses)
    return evaluate_losses, return_metrics
