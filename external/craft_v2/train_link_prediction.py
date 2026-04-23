import logging
import time
import sys
import os
import numpy as np
import warnings
import json
import torch
import torch.nn as nn

from models.TGAT import TGAT
from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.CAWN import CAWN
from models.TCL import TCL
from models.GraphMixer import GraphMixer
from models.DyGFormer import DyGFormer
from models.CRAFT import CRAFT
from models.TPNet import TPNet, RandomProjectionModule
from models.dual_path_predictor import DualPathLinkPredictor
from models.modules import MergeLayer, MulMergeLayer, BPRLoss
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler
from evaluate_models_utils import evaluate_model_link_prediction
from evaluate_models_utils import evaluate_model_link_prediction_multi_negs
from utils.metrics import get_link_prediction_metrics
from utils.DataLoader import get_idx_data_loader, get_link_prediction_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args
from utils.temporal_stats_bank import TemporalStatsBank
from models.SASRec import SASRec
from models.SGNNHN import SGNNHN


PROJECTION_BASED_MODELS = ['TPNet']
PAS_MODELS = ['TPNet', 'CRAFT']


def is_craft_pas_enabled(args, model: nn.Module = None):
    return args.model_name == 'CRAFT'


def flush_logger_handlers(logger: logging.Logger):
    if logger is None:
        return
    for handler in logger.handlers:
        try:
            handler.flush()
        except Exception:
            continue


def maybe_log_loop_progress(logger: logging.Logger, stage: str, epoch: int, batch_idx: int, num_batches: int,
                            stage_start_time: float, last_log_time: float, interval_seconds: float = 300.0,
                            extra_message: str = ''):
    if logger is None:
        return last_log_time
    now = time.time()
    should_log = batch_idx == 0 or (batch_idx + 1) == num_batches or (now - last_log_time) >= interval_seconds
    if not should_log:
        return last_log_time
    message = f'{stage} epoch {epoch + 1}: batch {batch_idx + 1}/{num_batches}, elapsed {now - stage_start_time:.1f}s'
    if extra_message:
        message = f'{message}, {extra_message}'
    logger.info(message)
    flush_logger_handlers(logger)
    return now


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


def create_tpnet_stats_bank(train_data, node_raw_features: np.ndarray, args, seed: int):
    return TemporalStatsBank(
        num_nodes=node_raw_features.shape[0],
        all_dst_node_ids=np.unique(train_data.dst_node_ids),
        rho_ema_momentum=args.rho_ema_momentum,
        seed=seed,
    )


def preload_tpnet_stats_bank(stats_bank: TemporalStatsBank, data):
    stats_bank.reset()
    stats_bank.update_batch(src=data.src_node_ids, dst=data.dst_node_ids, t=data.node_interact_times)
    return stats_bank


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


def compute_craft_edge_embeddings(backbone: nn.Module, model_inputs: dict):
    edge_embeddings = backbone.compute_src_dst_edge_embeddings(**model_inputs)
    edge_embeddings = edge_embeddings.reshape(-1, edge_embeddings.shape[-1])
    edge_embeddings = torch.nan_to_num(edge_embeddings, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
    return edge_embeddings


def compute_craft_scores(predictor: nn.Module, craft_embeddings: torch.Tensor, entity_stats: np.ndarray,
                         pair_stats: np.ndarray, lambda_pair: float):
    craft_embeddings = craft_embeddings.reshape(-1, craft_embeddings.shape[-1])
    entity_stats = torch.from_numpy(entity_stats).to(device=craft_embeddings.device, dtype=craft_embeddings.dtype)
    pair_stats = torch.from_numpy(pair_stats).to(device=craft_embeddings.device, dtype=craft_embeddings.dtype)
    entity_input = torch.cat([craft_embeddings, entity_stats], dim=-1)
    pair_input = torch.cat([craft_embeddings, pair_stats], dim=-1)
    s_entity = predictor.entity_head(entity_input).squeeze(dim=-1)
    s_pair = predictor.pair_head(pair_input).squeeze(dim=-1)
    return s_entity, s_pair, s_entity + lambda_pair * s_pair


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


def train_epoch(model, args, logger, epoch, train_idx_data_loader, train_neighbor_sampler, train_neg_edge_sampler, train_data, optimizer, loss_func, full_neighbor_sampler, val_data, val_idx_data_loader, val_neg_edge_sampler, full_data, stats_bank=None):
        model.train()
        craft_pas_enabled = is_craft_pas_enabled(args, model)
        if args.model_name not in ['CRAFT']:
            model[0].set_neighbor_sampler(train_neighbor_sampler)
        epoch_start_time = time.time()
        num_train_batches = len(train_idx_data_loader)
        logger.info(f'Epoch {epoch + 1} started with {num_train_batches} train batches.')
        flush_logger_handlers(logger)
        train_progress_log_time = epoch_start_time
        
        if args.model_name in ['JODIE', 'DyRep', 'TGN']:
            # reinitialize memory of memory-based models at the start of each epoch
            model[0].memory_bank.__init_memory_bank__()
        if craft_pas_enabled:
            if stats_bank is None:
                raise ValueError('CRAFT training requires a TemporalStatsBank instance!')
            stats_bank.reset()
            craft_pas_bce_loss_func = nn.BCEWithLogitsLoss()
        if args.model_name in PROJECTION_BASED_MODELS:
            if stats_bank is None:
                raise ValueError('TPNet training requires a TemporalStatsBank instance!')
            stats_bank.reset()
            model[0].random_projections.reset_random_projections()
            tpnet_loss_func = nn.BCEWithLogitsLoss(reduction='none')
        # store train losses and metrics
        train_losses, train_metrics = [], []
        for batch_idx, train_data_indices in enumerate(train_idx_data_loader):
            train_data_indices = train_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], \
                train_data.node_interact_times[train_data_indices], train_data.edge_ids[train_data_indices]
            if craft_pas_enabled:
                if args.collision_check:
                    batch_neg_random_dst_node_ids = train_neg_edge_sampler.sample_with_time_collision_check(
                        num_negs=1,
                        batch_src_node_ids=batch_src_node_ids,
                        batch_node_interact_times=batch_node_interact_times,
                        neighbor_sampler=train_neighbor_sampler,
                    ).flatten()
                else:
                    _, batch_neg_random_dst_node_ids = train_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                batch_neg_hard_dst_node_ids = stats_bank.sample_hard_neg(src=batch_src_node_ids, dst=batch_dst_node_ids, n=1)

                shared_craft_inputs = create_craft_src_history_inputs(
                    neighbor_sampler=train_neighbor_sampler,
                    src_node_ids=batch_src_node_ids,
                    node_interact_times=batch_node_interact_times,
                    num_neighbors=args.num_neighbors,
                )
                if shared_craft_inputs['src_neighb_seq_len'].sum().item() == 0:
                    stats_bank.update_batch(src=batch_src_node_ids, dst=batch_dst_node_ids, t=batch_node_interact_times)
                    continue
                batch_pos_inputs = attach_craft_candidate_inputs(
                    neighbor_sampler=train_neighbor_sampler,
                    shared_inputs=shared_craft_inputs,
                    dst_node_ids=batch_dst_node_ids,
                    node_interact_times=batch_node_interact_times,
                )
                batch_random_neg_inputs = attach_craft_candidate_inputs(
                    neighbor_sampler=train_neighbor_sampler,
                    shared_inputs=shared_craft_inputs,
                    dst_node_ids=batch_neg_random_dst_node_ids,
                    node_interact_times=batch_node_interact_times,
                )
                batch_hard_neg_inputs = attach_craft_candidate_inputs(
                    neighbor_sampler=train_neighbor_sampler,
                    shared_inputs=shared_craft_inputs,
                    dst_node_ids=batch_neg_hard_dst_node_ids,
                    node_interact_times=batch_node_interact_times,
                )

                entity_stats_pos, pair_stats_pos = stats_bank.query(src=batch_src_node_ids, dst=batch_dst_node_ids, t=batch_node_interact_times)
                entity_stats_random_neg, pair_stats_random_neg = stats_bank.query(
                    src=batch_src_node_ids,
                    dst=batch_neg_random_dst_node_ids,
                    t=batch_node_interact_times,
                )
                entity_stats_hard_neg, pair_stats_hard_neg = stats_bank.query(
                    src=batch_src_node_ids,
                    dst=batch_neg_hard_dst_node_ids,
                    t=batch_node_interact_times,
                )

                model[1].reset_cached_projections()

                batch_pos_embeddings = compute_craft_edge_embeddings(
                    backbone=model[0],
                    model_inputs=batch_pos_inputs,
                ).squeeze(dim=1)
                batch_random_neg_embeddings = compute_craft_edge_embeddings(
                    backbone=model[0],
                    model_inputs=batch_random_neg_inputs,
                ).squeeze(dim=1)
                batch_hard_neg_embeddings = compute_craft_edge_embeddings(
                    backbone=model[0],
                    model_inputs=batch_hard_neg_inputs,
                ).squeeze(dim=1)

                s_ent_pos, s_pair_pos, s_total_pos = compute_craft_scores(
                    predictor=model[1],
                    craft_embeddings=batch_pos_embeddings,
                    entity_stats=entity_stats_pos,
                    pair_stats=pair_stats_pos,
                    lambda_pair=args.lambda_pair,
                )
                s_ent_random_neg, s_pair_random_neg, s_total_random_neg = compute_craft_scores(
                    predictor=model[1],
                    craft_embeddings=batch_random_neg_embeddings,
                    entity_stats=entity_stats_random_neg,
                    pair_stats=pair_stats_random_neg,
                    lambda_pair=args.lambda_pair,
                )
                s_ent_hard_neg, s_pair_hard_neg, _ = compute_craft_scores(
                    predictor=model[1],
                    craft_embeddings=batch_hard_neg_embeddings,
                    entity_stats=entity_stats_hard_neg,
                    pair_stats=pair_stats_hard_neg,
                    lambda_pair=args.lambda_pair,
                )

                if args.training_mode == 'joint_random':
                    l_ent = compute_pas_loss(s_ent_pos, s_ent_random_neg, args.loss, loss_func, craft_pas_bce_loss_func)
                    l_pair = compute_pas_loss(s_pair_pos, s_pair_random_neg, args.loss, loss_func, craft_pas_bce_loss_func)
                    loss = l_ent + args.lambda_pair * l_pair
                elif args.training_mode == 'joint_hard':
                    l_ent = compute_pas_loss(s_ent_pos, s_ent_hard_neg, args.loss, loss_func, craft_pas_bce_loss_func)
                    l_pair = compute_pas_loss(s_pair_pos, s_pair_hard_neg, args.loss, loss_func, craft_pas_bce_loss_func)
                    loss = l_ent + args.lambda_pair * l_pair
                elif args.training_mode == 'decoupled':
                    l_ent = compute_pas_loss(s_ent_pos, s_ent_random_neg, args.loss, loss_func, craft_pas_bce_loss_func)
                    l_pair = compute_pas_loss(s_pair_pos, s_pair_hard_neg, args.loss, loss_func, craft_pas_bce_loss_func)
                    loss = l_ent + args.lambda_pair * l_pair
                elif args.training_mode == 'decoupled_stopgrad':
                    l_ent = compute_pas_loss(s_ent_pos, s_ent_random_neg, args.loss, loss_func, craft_pas_bce_loss_func)
                    _, s_pair_pos_sg, _ = compute_craft_scores(
                        predictor=model[1],
                        craft_embeddings=batch_pos_embeddings.detach(),
                        entity_stats=entity_stats_pos,
                        pair_stats=pair_stats_pos,
                        lambda_pair=args.lambda_pair,
                    )
                    _, s_pair_hard_neg_sg, _ = compute_craft_scores(
                        predictor=model[1],
                        craft_embeddings=batch_hard_neg_embeddings.detach(),
                        entity_stats=entity_stats_hard_neg,
                        pair_stats=pair_stats_hard_neg,
                        lambda_pair=args.lambda_pair,
                    )
                    l_pair = compute_pas_loss(s_pair_pos_sg, s_pair_hard_neg_sg, args.loss, loss_func, craft_pas_bce_loss_func)
                    loss = l_ent + args.lambda_pair * l_pair
                else:
                    raise ValueError(f"Unsupported CRAFT training mode {args.training_mode}!")

                if args.lambda_orth > 0:
                    loss = loss + args.lambda_orth * model[1].orthogonality_loss()

                positive_probabilities = format_pas_predicts(s_total_pos, args.loss)
                negative_probabilities = format_pas_predicts(s_total_random_neg, args.loss)
                predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
                labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)
                train_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))
                train_losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                stats_bank.update_batch(src=batch_src_node_ids, dst=batch_dst_node_ids, t=batch_node_interact_times)
                train_progress_log_time = maybe_log_loop_progress(
                    logger=logger,
                    stage='train',
                    epoch=epoch,
                    batch_idx=batch_idx,
                    num_batches=num_train_batches,
                    stage_start_time=epoch_start_time,
                    last_log_time=train_progress_log_time,
                    extra_message=f'loss={loss.item():.6f}',
                )
                continue
            if args.model_name in PROJECTION_BASED_MODELS:
                if args.collision_check:
                    batch_neg_random_dst_node_ids = train_neg_edge_sampler.sample_with_time_collision_check(
                        num_negs=1,
                        batch_src_node_ids=batch_src_node_ids,
                        batch_node_interact_times=batch_node_interact_times,
                        neighbor_sampler=train_neighbor_sampler,
                    ).flatten()
                else:
                    _, batch_neg_random_dst_node_ids = train_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                batch_neg_random_src_node_ids = batch_src_node_ids
                batch_neg_hard_dst_node_ids = stats_bank.sample_hard_neg(src=batch_src_node_ids, dst=batch_dst_node_ids, n=1)

                entity_stats_pos, pair_stats_pos = stats_bank.query(src=batch_src_node_ids, dst=batch_dst_node_ids, t=batch_node_interact_times)
                entity_stats_random_neg, pair_stats_random_neg = stats_bank.query(
                    src=batch_neg_random_src_node_ids,
                    dst=batch_neg_random_dst_node_ids,
                    t=batch_node_interact_times,
                )
                entity_stats_hard_neg, pair_stats_hard_neg = stats_bank.query(
                    src=batch_src_node_ids,
                    dst=batch_neg_hard_dst_node_ids,
                    t=batch_node_interact_times,
                )

                model[1].reset_cached_projections()

                batch_src_node_embeddings, batch_dst_node_embeddings = compute_tpnet_temporal_embeddings(
                    backbone=model[0],
                    src_node_ids=batch_src_node_ids,
                    dst_node_ids=batch_dst_node_ids,
                    node_interact_times=batch_node_interact_times,
                )
                batch_neg_random_src_node_embeddings, batch_neg_random_dst_node_embeddings = compute_tpnet_temporal_embeddings(
                    backbone=model[0],
                    src_node_ids=batch_neg_random_src_node_ids,
                    dst_node_ids=batch_neg_random_dst_node_ids,
                    node_interact_times=batch_node_interact_times,
                )
                batch_neg_hard_src_node_embeddings, batch_neg_hard_dst_node_embeddings = compute_tpnet_temporal_embeddings(
                    backbone=model[0],
                    src_node_ids=batch_src_node_ids,
                    dst_node_ids=batch_neg_hard_dst_node_ids,
                    node_interact_times=batch_node_interact_times,
                )

                s_ent_pos, s_pair_pos, s_total_pos = compute_tpnet_scores(
                    predictor=model[1],
                    src_embeddings=batch_src_node_embeddings,
                    dst_embeddings=batch_dst_node_embeddings,
                    entity_stats=entity_stats_pos,
                    pair_stats=pair_stats_pos,
                )
                s_ent_random_neg, s_pair_random_neg, s_total_random_neg = compute_tpnet_scores(
                    predictor=model[1],
                    src_embeddings=batch_neg_random_src_node_embeddings,
                    dst_embeddings=batch_neg_random_dst_node_embeddings,
                    entity_stats=entity_stats_random_neg,
                    pair_stats=pair_stats_random_neg,
                )
                s_ent_hard_neg, s_pair_hard_neg, _ = compute_tpnet_scores(
                    predictor=model[1],
                    src_embeddings=batch_neg_hard_src_node_embeddings,
                    dst_embeddings=batch_neg_hard_dst_node_embeddings,
                    entity_stats=entity_stats_hard_neg,
                    pair_stats=pair_stats_hard_neg,
                )

                if args.training_mode == 'joint_random':
                    l_ent = (tpnet_loss_func(input=s_ent_pos, target=torch.ones_like(s_ent_pos)).mean() +
                             tpnet_loss_func(input=s_ent_random_neg, target=torch.zeros_like(s_ent_random_neg)).mean()) / 2.0
                    l_pair = (tpnet_loss_func(input=s_pair_pos, target=torch.ones_like(s_pair_pos)).mean() +
                              tpnet_loss_func(input=s_pair_random_neg, target=torch.zeros_like(s_pair_random_neg)).mean()) / 2.0
                    loss = l_ent + args.lambda_pair * l_pair
                elif args.training_mode == 'joint_hard':
                    l_ent = (tpnet_loss_func(input=s_ent_pos, target=torch.ones_like(s_ent_pos)).mean() +
                             tpnet_loss_func(input=s_ent_hard_neg, target=torch.zeros_like(s_ent_hard_neg)).mean()) / 2.0
                    l_pair = (tpnet_loss_func(input=s_pair_pos, target=torch.ones_like(s_pair_pos)).mean() +
                              tpnet_loss_func(input=s_pair_hard_neg, target=torch.zeros_like(s_pair_hard_neg)).mean()) / 2.0
                    loss = l_ent + args.lambda_pair * l_pair
                elif args.training_mode == 'decoupled':
                    l_ent = (tpnet_loss_func(input=s_ent_pos, target=torch.ones_like(s_ent_pos)).mean() +
                             tpnet_loss_func(input=s_ent_random_neg, target=torch.zeros_like(s_ent_random_neg)).mean()) / 2.0
                    l_pair = (tpnet_loss_func(input=s_pair_pos, target=torch.ones_like(s_pair_pos)).mean() +
                              tpnet_loss_func(input=s_pair_hard_neg, target=torch.zeros_like(s_pair_hard_neg)).mean()) / 2.0
                    loss = l_ent + args.lambda_pair * l_pair
                elif args.training_mode == 'decoupled_stopgrad':
                    l_ent = (tpnet_loss_func(input=s_ent_pos, target=torch.ones_like(s_ent_pos)).mean() +
                             tpnet_loss_func(input=s_ent_random_neg, target=torch.zeros_like(s_ent_random_neg)).mean()) / 2.0
                    _, s_pair_pos_sg, _ = compute_tpnet_scores(
                        predictor=model[1],
                        src_embeddings=batch_src_node_embeddings.detach(),
                        dst_embeddings=batch_dst_node_embeddings.detach(),
                        entity_stats=entity_stats_pos,
                        pair_stats=pair_stats_pos,
                    )
                    _, s_pair_hard_neg_sg, _ = compute_tpnet_scores(
                        predictor=model[1],
                        src_embeddings=batch_neg_hard_src_node_embeddings.detach(),
                        dst_embeddings=batch_neg_hard_dst_node_embeddings.detach(),
                        entity_stats=entity_stats_hard_neg,
                        pair_stats=pair_stats_hard_neg,
                    )
                    l_pair = (tpnet_loss_func(input=s_pair_pos_sg, target=torch.ones_like(s_pair_pos_sg)).mean() +
                              tpnet_loss_func(input=s_pair_hard_neg_sg, target=torch.zeros_like(s_pair_hard_neg_sg)).mean()) / 2.0
                    loss = l_ent + args.lambda_pair * l_pair
                else:
                    raise ValueError(f"Unsupported TPNet training mode {args.training_mode}!")

                if args.lambda_orth > 0:
                    loss = loss + args.lambda_orth * model[1].orthogonality_loss()

                positive_probabilities = torch.sigmoid(s_total_pos.clamp(-30, 30))
                negative_probabilities = torch.sigmoid(s_total_random_neg.clamp(-30, 30))
                predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
                labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)
                train_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))
                train_losses.append(loss.item())

                model[0].random_projections.update(
                    src_node_ids=batch_src_node_ids,
                    dst_node_ids=batch_dst_node_ids,
                    node_interact_times=batch_node_interact_times,
                )

                optimizer.zero_grad()
                loss.backward()
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                stats_bank.update_batch(src=batch_src_node_ids, dst=batch_dst_node_ids, t=batch_node_interact_times)
                train_progress_log_time = maybe_log_loop_progress(
                    logger=logger,
                    stage='train',
                    epoch=epoch,
                    batch_idx=batch_idx,
                    num_batches=num_train_batches,
                    stage_start_time=epoch_start_time,
                    last_log_time=train_progress_log_time,
                    extra_message=f'loss={loss.item():.6f}',
                )
                continue
            if args.collision_check: 
                batch_neg_dst_node_ids = train_neg_edge_sampler.sample_with_time_collision_check(num_negs=1, batch_src_node_ids=batch_src_node_ids, batch_node_interact_times=batch_node_interact_times, neighbor_sampler=train_neighbor_sampler).flatten()
                batch_neg_src_node_ids = batch_src_node_ids
            else:
                batch_neg_src_node_ids, batch_neg_dst_node_ids = train_neg_edge_sampler.sample(size=len(batch_src_node_ids))
            
            # we need to compute for positive and negative edges respectively, because the new sampling strategy (for evaluation) allows the negative source nodes to be
            # different from the source nodes, this is different from previous works that just replace destination nodes with negative destination nodes
            if args.model_name in ['TGAT', 'CAWN', 'TCL']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                        dst_node_ids=batch_dst_node_ids,
                                                                        node_interact_times=batch_node_interact_times,
                                                                        num_neighbors=args.num_neighbors)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                        dst_node_ids=batch_neg_dst_node_ids,
                                                                        node_interact_times=batch_node_interact_times,
                                                                        num_neighbors=args.num_neighbors)
            elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
                # note that negative nodes do not change the memories while the positive nodes change the memories,
                # we need to first compute the embeddings of negative nodes for memory-based models
                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_node_interact_times=np.concatenate([batch_node_interact_times,batch_node_interact_times,batch_node_interact_times],axis=0)
                batch_src_node_embeddings, batch_dst_node_embeddings, batch_neg_dst_node_embeddings = \
                model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                    dst_node_ids=batch_dst_node_ids,
                                                                    neg_dst_node_ids=batch_neg_dst_node_ids,
                                                                    node_interact_times=batch_node_interact_times,
                                                                    edge_ids=batch_edge_ids,
                                                                    num_neighbors=args.num_neighbors)
                batch_neg_src_node_embeddings = batch_src_node_embeddings
            elif args.model_name in ['GraphMixer']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                        dst_node_ids=batch_dst_node_ids,
                                                                        node_interact_times=batch_node_interact_times,
                                                                        num_neighbors=args.num_neighbors,
                                                                        time_gap=args.time_gap)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                        dst_node_ids=batch_neg_dst_node_ids,
                                                                        node_interact_times=batch_node_interact_times,
                                                                        num_neighbors=args.num_neighbors,
                                                                        time_gap=args.time_gap)
            elif args.model_name in ['DyGFormer']:
                src_node_ids = np.concatenate([batch_src_node_ids, batch_neg_src_node_ids], axis=0)
                dst_node_ids = np.concatenate([batch_dst_node_ids, batch_neg_dst_node_ids], axis=0)
                node_interact_times = np.concatenate([batch_node_interact_times, batch_node_interact_times], axis=0)
                src_node_embeddings, dst_node_embeddings = model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times)
                batch_src_node_embeddings, batch_neg_src_node_embeddings = src_node_embeddings[:len(batch_src_node_ids)], src_node_embeddings[len(batch_src_node_ids):]
                batch_dst_node_embeddings, batch_neg_dst_node_embeddings = dst_node_embeddings[:len(batch_dst_node_ids)], dst_node_embeddings[len(batch_dst_node_ids):]
            elif args.model_name in ['SASRec', 'SGNNHN']:
                neighbor_node_ids, _, _=train_neighbor_sampler.get_historical_neighbors_left(node_ids=batch_src_node_ids, node_interact_times=batch_node_interact_times, num_neighbors=args.num_neighbors)
                neighbor_num=(neighbor_node_ids!=0).sum(axis=1)
                if neighbor_num.sum() == 0:
                    continue
                pos_item = torch.from_numpy(batch_dst_node_ids)
                neg_item = torch.from_numpy(batch_neg_dst_node_ids)
                test_items = torch.cat([pos_item, neg_item], dim=0)
                batch_data=[torch.from_numpy(neighbor_node_ids), torch.from_numpy(neighbor_num), test_items]
                batch_src_node_embeddings, dst_node_embeddings = model[0].calculate_loss(batch_data)
                batch_dst_node_embeddings = dst_node_embeddings[:len(pos_item)]
                batch_neg_dst_node_embeddings = dst_node_embeddings[len(pos_item):]
                batch_neg_src_node_embeddings = batch_src_node_embeddings
            elif args.model_name in ['CRAFT']:
                src_neighb_seq, _, src_neighb_interact_times = train_neighbor_sampler.get_historical_neighbors_left(node_ids=batch_src_node_ids, node_interact_times=batch_node_interact_times, num_neighbors=args.num_neighbors)
                neighbor_num=(src_neighb_seq!=0).sum(axis=1)
                if neighbor_num.sum() == 0:
                    continue
                pos_item = torch.from_numpy(batch_dst_node_ids).unsqueeze(-1)
                neg_item = torch.from_numpy(batch_neg_dst_node_ids).unsqueeze(-1)
                test_dst = torch.cat([pos_item, neg_item], dim=-1)
                dst_last_neighbor, _, dst_last_update_time = train_neighbor_sampler.get_historical_neighbors_left(node_ids=test_dst.flatten(), node_interact_times=np.broadcast_to(batch_node_interact_times[:,np.newaxis], (len(batch_node_interact_times), test_dst.shape[1])).flatten(), num_neighbors=1)
                dst_last_update_time = np.array(dst_last_update_time).reshape(len(test_dst), -1)
                dst_last_update_time[dst_last_neighbor.reshape(len(test_dst),-1)==0]=-100000
                dst_last_update_time = torch.from_numpy(dst_last_update_time)
                loss, predicts, labels = model.calculate_loss(src_neighb_seq=torch.from_numpy(src_neighb_seq), 
                                                                src_neighb_seq_len=torch.from_numpy(neighbor_num), 
                                                                src_neighb_interact_times=torch.from_numpy(src_neighb_interact_times), 
                                                                cur_pred_times=torch.from_numpy(batch_node_interact_times), 
                                                                test_dst=test_dst, 
                                                                dst_last_update_times=dst_last_update_time)
            else:
                raise ValueError(f"Wrong value for model_name {args.model_name}!")
            if args.model_name not in ['CRAFT']:
                if args.loss in ['BPR']:
                    positive_probabilities = model[1](
                        input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1)
                    negative_probabilities = model[1](
                        input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1)
                    loss = loss_func(positive_probabilities, negative_probabilities)
                    predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
                    labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)
                else: # default BCE
                    positive_probabilities = model[1](
                        input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1).sigmoid()
                    negative_probabilities = model[1](
                        input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1).sigmoid()
                    predicts = torch.cat(
                        [positive_probabilities, negative_probabilities], dim=0)
                    labels = torch.cat([torch.ones_like(
                        positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)
                    train_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))
                    loss = loss_func(input=predicts, target=labels)
            if predicts is not None and labels is not None:
                train_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))
            train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                # detach the memories and raw messages of nodes in the memory bank after each batch, so we don't back propagate to the start of time
                model[0].memory_bank.detach_memory_bank()
            train_progress_log_time = maybe_log_loop_progress(
                logger=logger,
                stage='train',
                epoch=epoch,
                batch_idx=batch_idx,
                num_batches=num_train_batches,
                stage_start_time=epoch_start_time,
                last_log_time=train_progress_log_time,
                extra_message=f'loss={loss.item():.6f}',
            )
        logger.info(f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {np.mean(train_losses):.8f}')
        if len(train_metrics)>0:
            for metric_name in train_metrics[0].keys():
                logger.info(f'train {metric_name}, {np.mean([train_metric[metric_name] for train_metric in train_metrics]):.8f}')
        logger.info(f'Epoch {epoch + 1} training finished in {time.time() - epoch_start_time:.1f}s, starting validation.')
        flush_logger_handlers(logger)
        if args.model_name in ['JODIE', 'DyRep', 'TGN', 'TGN_mem']:
            train_backup_memory_bank = model[0].memory_bank.backup_memory_bank()
        if args.model_name in PROJECTION_BASED_MODELS:
            tpnet_train_backup = model[0].random_projections.backup_random_projections()
        if args.use_mrr_val:
            val_metrics = evaluate_model_link_prediction_multi_negs(model_name=args.model_name,
                                                                    model=model,
                                                                    neighbor_sampler=full_neighbor_sampler,
                                                                    evaluate_idx_data_loader=val_idx_data_loader,
                                                                    evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                                                    evaluate_data=val_data,
                                                                    loss_func=loss_func,
                                                                    device=args.device,
                                                                    num_neighbors=args.num_neighbors,
                                                                    time_gap=args.time_gap, mode='val', loss_type = args.loss, full_data=full_data,  dataset_name=args.dataset_name, collision_check=args.collision_check, stats_bank=stats_bank, lambda_pair=args.lambda_pair, logger=logger)
        else:
            val_losses, val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                    model=model,
                                                                    neighbor_sampler=full_neighbor_sampler,
                                                                    evaluate_idx_data_loader=val_idx_data_loader,
                                                                    evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                                                    evaluate_data=val_data,
                                                                    loss_func=loss_func,
                                                                    device=args.device,
                                                                    num_neighbors=args.num_neighbors,
                                                                    time_gap=args.time_gap,mode='val', loss_type = args.loss, full_data=full_data, collision_check=args.collision_check, dataset_name=args.dataset_name, stats_bank=stats_bank, lambda_pair=args.lambda_pair, logger=logger)

        if args.model_name in ['JODIE', 'DyRep', 'TGN']:
            # ! This is different from the original DyGLib. We reload the training memory bank in order to store the training memory bank. When testing, we must deal with the edges in val set first.
            model[0].memory_bank.reload_memory_bank(train_backup_memory_bank)
        if args.model_name in PROJECTION_BASED_MODELS:
            model[0].random_projections.reload_random_projections(tpnet_train_backup)
        for metric_name in val_metrics.keys():
            logger.info(f'validate {metric_name}, {val_metrics[metric_name]:.8f}')

        flush_logger_handlers(logger)

        return val_metrics

def get_model(args, train_data, node_raw_features, edge_raw_features, train_neighbor_sampler, full_data,logger):
    if args.model_name in ['JODIE']:
        src_node_mean_time_shift, src_node_std_time_shift, dst_node_mean_time_shift_dst, dst_node_std_time_shift = \
            compute_src_dst_node_time_shifts(train_data.src_node_ids, train_data.dst_node_ids, train_data.node_interact_times)
    if args.model_name == 'TGAT':
        dynamic_backbone = TGAT(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, device=args.device)
    elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
        # four floats that represent the mean and standard deviation of source and destination node time shifts in the training data, which is used for JODIE
        src_node_mean_time_shift, src_node_std_time_shift, dst_node_mean_time_shift_dst, dst_node_std_time_shift = \
            compute_src_dst_node_time_shifts(train_data.src_node_ids, train_data.dst_node_ids, train_data.node_interact_times)
        dynamic_backbone = MemoryModel(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                        time_feat_dim=args.time_feat_dim, model_name=args.model_name, num_layers=args.num_layers, num_heads=args.num_heads,
                                        dropout=args.dropout, src_node_mean_time_shift=src_node_mean_time_shift, src_node_std_time_shift=src_node_std_time_shift,
                                        dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst, dst_node_std_time_shift=dst_node_std_time_shift, device=args.device)
    elif args.model_name == 'CAWN':
        dynamic_backbone = CAWN(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                time_feat_dim=args.time_feat_dim, position_feat_dim=args.position_feat_dim, walk_length=args.walk_length,
                                num_walk_heads=args.num_walk_heads, dropout=args.dropout, device=args.device)
    elif args.model_name == 'TCL':
        dynamic_backbone = TCL(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads,
                                num_depths=args.num_neighbors + 1, dropout=args.dropout, device=args.device)
    elif args.model_name == 'GraphMixer':
        dynamic_backbone = GraphMixer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                        time_feat_dim=args.time_feat_dim, num_tokens=args.num_neighbors, num_layers=args.num_layers, dropout=args.dropout, device=args.device)
    elif args.model_name == 'DyGFormer':
        dynamic_backbone = DyGFormer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                        time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                        num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                        max_input_sequence_length=args.max_input_sequence_length, device=args.device)
    elif args.model_name == 'TPNet':
        tpnet_feat_dim = args.output_dim
        if node_raw_features.shape[1] > tpnet_feat_dim:
            raise ValueError(f"TPNet expects node feature dim <= output_dim, got {node_raw_features.shape[1]} > {tpnet_feat_dim}")
        padded_node_raw_features = node_raw_features
        if padded_node_raw_features.shape[1] < tpnet_feat_dim:
            node_zero_padding = np.zeros((padded_node_raw_features.shape[0], tpnet_feat_dim - padded_node_raw_features.shape[1]))
            padded_node_raw_features = np.concatenate([padded_node_raw_features, node_zero_padding], axis=1)
        rp_module = RandomProjectionModule(
            node_num=padded_node_raw_features.shape[0],
            edge_num=edge_raw_features.shape[0],
            dim_factor=args.tpnet_dim_factor,
            num_layer=args.tpnet_num_layer,
            time_decay_weight=args.tpnet_time_decay,
            device=args.device,
            use_matrix=False,
            beginning_time=np.float64(np.min(train_data.node_interact_times)),
            not_scale=False,
            enforce_dim=-1,
        )
        dynamic_backbone = TPNet(node_raw_features=padded_node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                 time_feat_dim=args.time_feat_dim, dropout=args.dropout, random_projections=rp_module,
                                 num_layers=args.num_layers, num_neighbors=args.num_neighbors, device=args.device)
    elif args.model_name == 'SASRec':
        dynamic_backbone = SASRec(args.num_layers, args.num_heads, args.embedding_size, args.inner_size, args.hidden_dropout, args.attn_dropout_prob, args.hidden_act, args.layer_norm_eps, args.initializer_range, args.item_size, max_seq_length = args.num_neighbors, device=args.device)
    elif args.model_name == 'SGNNHN':
        dynamic_backbone = SGNNHN(args.embedding_size, args.step, args.device, args.scale, args.item_size, args.dropout, args.num_neighbors, loss_type=args.loss)
    elif args.model_name in ['CRAFT']:
        dynamic_backbone = CRAFT(args.num_layers, args.num_heads, args.embedding_size, args.hidden_dropout, args.attn_dropout_prob, args.hidden_act, args.layer_norm_eps, args.initializer_range, args.item_size, max_seq_length = args.num_neighbors, device=args.device, loss_type=args.loss, use_pos=args.use_pos, input_cat_time_intervals=args.input_cat_time_intervals, output_cat_time_intervals=args.output_cat_time_intervals, output_cat_repeat_times=args.output_cat_repeat_times, num_output_layer=args.num_output_layer, emb_dropout_prob=args.emb_dropout_prob, skip_connection=args.skip_connection)
    else:
        raise ValueError(f"Wrong value for model_name {args.model_name}!")
    if args.model_name == 'TPNet':
        link_predictor = DualPathLinkPredictor(input_dim=args.output_dim, hidden_dim=args.output_dim)
    elif args.model_name == 'CRAFT' and args.use_pas:
        link_predictor = DualPathLinkPredictor(input_dim=dynamic_backbone.hidden_size,
                                               hidden_dim=dynamic_backbone.hidden_size)
    elif args.merge in ['cat']:
        link_predictor = MergeLayer(input_dim1=args.output_dim, input_dim2=args.output_dim, hidden_dim=args.output_dim, output_dim=1)
    elif args.merge in ['mul']:
        link_predictor = MulMergeLayer(scale=args.scale)
    else:
        link_predictor = None
    if args.model_name in ['CRAFT', 'SASRec', 'SGNNHN']:
        dynamic_backbone.set_min_idx(src_min_idx=args.src_min_idx, dst_min_idx=args.dst_min_idx)
    if args.model_name not in ['CRAFT'] or args.use_pas:
        model = nn.Sequential(dynamic_backbone, link_predictor)
    else:
        model = dynamic_backbone
    logger.info(f'model -> {model}')
    logger.info(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
                f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')
    return model

def get_loss_fn(args):
    if args.loss in ['BPR']:
        loss_func = BPRLoss()
    elif args.loss in ['BCE']:
        loss_func = nn.BCELoss()
    else:
        raise NotImplementedError(f"Loss function {args.loss} not implemented!")
    return loss_func

if __name__ == "__main__":

    warnings.filterwarnings('ignore')
    args = get_link_prediction_args(is_evaluation=False)
    pas_mode_active = args.model_name in PAS_MODELS
    if args.model_name == 'TPNet' and args.loss != 'BCE':
        raise ValueError('TPNet+PAS currently supports BCE loss only.')
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    postfix = ''
    if args.use_edge_feat:
        postfix += '_e'
    if args.use_node_feat:
        postfix+='_n'
    mode_postfix = f'_{args.training_mode}' if pas_mode_active else ''
    if args.version is not None:
        args.save_model_name = f'{args.model_name}_seed{args.seed}_batchsize{args.batch_size}_num_neighbors{args.num_neighbors}_dropout{args.dropout}_sample_neighbor_strategy{args.sample_neighbor_strategy}_numlayers{args.num_layers}{postfix}{mode_postfix}_v{args.version}'
    else:
        args.save_model_name = f'{args.model_name}_seed{args.seed}_batchsize{args.batch_size}_num_neighbors{args.num_neighbors}_dropout{args.dropout}_sample_neighbor_strategy{args.sample_neighbor_strategy}_numlayers{args.num_layers}{postfix}{mode_postfix}'
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    log_dir = f"{current_dir}/logs/{args.dataset_name}/{args.model_name}/{args.version}_{args.save_model_name}"
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/{str(time.time())}.log"
    print("log in: ", log_file)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    fh.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(fh)
    # get data for training, validation and testing
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, val_test_data = \
        get_link_prediction_data(
            dataset_name=args.dataset_name, dataset_path=args.dataset_path, use_edge_feat=args.use_edge_feat, use_node_feat=args.use_node_feat, logger=logger, model_name=args.model_name)

    # initialize training neighbor sampler to retrieve temporal graph
    train_neighbor_sampler = get_neighbor_sampler(data=train_data, sample_neighbor_strategy=args.sample_neighbor_strategy,time_scaling_factor=args.time_scaling_factor, seed=0)

    # initialize validation and test neighbor sample to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,time_scaling_factor=args.time_scaling_factor, seed=1)
        
    if (args.is_bipartite or args.dataset_name in ['GoogleLocal', 'ML-20M', 'Taobao', 'Yelp', 'mooc', 'lastfm', 'reddit', 'wikipedia']):
        args.user_size = full_data.src_node_ids.max()-full_data.src_node_ids.min()+1
        args.item_size = full_data.dst_node_ids.max()-full_data.dst_node_ids.min()+1
        args.node_size = args.user_size + args.item_size
        args.dst_min_idx = full_data.dst_node_ids.min()
        args.src_min_idx = full_data.src_node_ids.min()
    else:
        args.user_size = full_data.max_node_id
        args.item_size = full_data.max_node_id
        args.node_size = args.user_size
        args.dst_min_idx = 1
        args.src_min_idx = 1
    
    # initialize negative samplers, set seeds for validation and testing so negatives are the same across different runs
    train_neg_edge_sampler = NegativeEdgeSampler(
        src_node_ids=train_data.src_node_ids, dst_node_ids=train_data.dst_node_ids, seed=0)
    val_neg_edge_sampler = NegativeEdgeSampler(
        src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=1)
    test_neg_edge_sampler = NegativeEdgeSampler(
        src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=2)
    if args.dataset_name in ['wikipedia', 'reddit', 'mooc', 'lastfm', 'uci', 'Flights' ]: # dataset with a small number of nodes
        args.collision_check = True
    if args.model_name in ['JODIE', 'DyRep', 'TGN']:
        test_data = val_test_data
        val_data = val_test_data
    train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(train_data.src_node_ids))), batch_size=args.batch_size, shuffle=args.shuffle)
    val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    # new_node_val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    test_idx_data_loader = get_idx_data_loader(indices_list=list(
        range(len(test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    multi_negs_test_idx_data_loader = get_idx_data_loader(indices_list=list(range(
        len(test_data.src_node_ids))), batch_size=args.multi_negs_batch_size, shuffle=False)
    # new_node_test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)

    val_metric_all_runs, test_metric_all_runs = [], []
    for run in range(args.num_runs):
        set_random_seed(seed=args.seed+run)
        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")
        logger.info(f'configuration is {args}')
        logger.info(f'{sys.argv}')
        model = get_model(args, train_data, node_raw_features, edge_raw_features, train_neighbor_sampler, full_data, logger)
        loss_func = get_loss_fn(args)
        optimizer = create_optimizer(model=model, optimizer_name=args.optimizer,
                                        learning_rate=args.learning_rate, weight_decay=args.weight_decay)
        # ! rename the save_model_name as seed have to change if num_runs > 1
        if args.version is not None:
            args.save_model_name = f'{args.model_name}_seed{args.seed+run}_batchsize{args.batch_size}_num_neighbors{args.num_neighbors}_dropout{args.dropout}_numlayers{args.num_layers}{postfix}{mode_postfix}_v{args.version}'
        else:
            args.save_model_name = f'{args.model_name}_seed{args.seed+run}_batchsize{args.batch_size}_num_neighbors{args.num_neighbors}_dropout{args.dropout}_numlayers{args.num_layers}{postfix}{mode_postfix}'
        save_model_folder = f"{args.save_model_path}/{args.dataset_name}/{args.model_name}/{args.save_model_name}/"
        if not os.path.exists(save_model_folder):
            os.makedirs(save_model_folder, exist_ok=True)

        early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder,save_model_name=args.save_model_name, logger=logger, model_name=args.model_name)
        if args.load_pretrained:
            early_stopping.load_checkpoint(model, map_location='cpu')
        model = convert_to_gpu(model, device=args.device)
        pas_stats_bank = create_tpnet_stats_bank(
            train_data=train_data,
            node_raw_features=node_raw_features,
            args=args,
            seed=args.seed + run,
        ) if pas_mode_active else None

        for epoch in range(args.num_epochs):
            val_metrics = train_epoch(model, args, logger, epoch, train_idx_data_loader, train_neighbor_sampler, train_neg_edge_sampler, train_data, optimizer, loss_func, full_neighbor_sampler, val_data, val_idx_data_loader, val_neg_edge_sampler, full_data, stats_bank=pas_stats_bank)
            if 'mrr' in val_metrics:
                val_metric_indicator = [('mrr', val_metrics['mrr'], True)]
            elif 'average_precision' in val_metrics:
                val_metric_indicator = [('average_precision', val_metrics['average_precision'], True)]
            else:
                raise ValueError(f"No valid metric found in val_metrics: {val_metrics}")
            early_stop = early_stopping.step(val_metric_indicator, model)
            if early_stop:
                break
            
        # load the best model
        early_stopping.load_checkpoint(model, map_location='cpu')
        model = convert_to_gpu(model, device=args.device)
        # evaluate the best model
        logger.info(f'get final performance on dataset {args.dataset_name}...')
        test_metrics={}
        if pas_mode_active:
            pas_eval_stats_bank = preload_tpnet_stats_bank(
                stats_bank=create_tpnet_stats_bank(train_data=train_data, node_raw_features=node_raw_features, args=args, seed=args.seed + run),
                data=train_data,
            )
            evaluate_model_link_prediction(model_name=args.model_name,
                                           model=model,
                                           neighbor_sampler=full_neighbor_sampler,
                                           evaluate_idx_data_loader=val_idx_data_loader,
                                           evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                           evaluate_data=val_data,
                                           loss_func=loss_func,
                                           device=args.device,
                                           num_neighbors=args.num_neighbors,
                                           time_gap=args.time_gap, mode='val', loss_type=args.loss, full_data=full_data,
                                           collision_check=args.collision_check, dataset_name=args.dataset_name, stats_bank=pas_eval_stats_bank, lambda_pair=args.lambda_pair, logger=logger)
            test_losses, test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                       model=model,
                                                                       neighbor_sampler=full_neighbor_sampler,
                                                                       evaluate_idx_data_loader=test_idx_data_loader,
                                                                       evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                       evaluate_data=test_data,
                                                                       loss_func=loss_func,
                                                                       device=args.device,
                                                                       num_neighbors=args.num_neighbors,
                                                                       time_gap=args.time_gap, mode='test', loss_type=args.loss, full_data=full_data,
                                                                       collision_check=args.collision_check, dataset_name=args.dataset_name, stats_bank=pas_eval_stats_bank, lambda_pair=args.lambda_pair, logger=logger)
            early_stopping.load_checkpoint(model, map_location='cpu')
            model = convert_to_gpu(model, device=args.device)
            pas_eval_stats_bank = preload_tpnet_stats_bank(
                stats_bank=create_tpnet_stats_bank(train_data=train_data, node_raw_features=node_raw_features, args=args, seed=args.seed + run),
                data=train_data,
            )
            evaluate_model_link_prediction_multi_negs(model_name=args.model_name,
                                                      model=model,
                                                      neighbor_sampler=full_neighbor_sampler,
                                                      evaluate_idx_data_loader=val_idx_data_loader,
                                                      evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                                      evaluate_data=val_data,
                                                      loss_func=loss_func,
                                                      device=args.device,
                                                      num_neighbors=args.num_neighbors,
                                                      time_gap=args.time_gap,
                                                      mode='val',
                                                      loss_type=args.loss,
                                                      full_data=full_data,
                                                      dataset_name=args.dataset_name,
                                                      collision_check=args.collision_check,
                                                      stats_bank=pas_eval_stats_bank,
                                                      lambda_pair=args.lambda_pair,
                                                      logger=logger)
            test_metrics_multi_negs = evaluate_model_link_prediction_multi_negs(
                model_name=args.model_name,
                model=model,
                neighbor_sampler=full_neighbor_sampler,
                evaluate_idx_data_loader=multi_negs_test_idx_data_loader,
                evaluate_neg_edge_sampler=test_neg_edge_sampler,
                evaluate_data=test_data,
                loss_func=loss_func,
                device=args.device,
                num_neighbors=args.num_neighbors,
                time_gap=args.time_gap,
                mode='test',
                loss_type=args.loss,
                full_data=full_data,
                dataset_name=args.dataset_name,
                collision_check=args.collision_check,
                stats_bank=pas_eval_stats_bank,
                lambda_pair=args.lambda_pair,
                logger=logger,
            )
        else:
            # For memory based models, we need to deal with their val set first in the evaluate_model_link_prediction function.
            test_losses, test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                       model=model,
                                                                       neighbor_sampler=full_neighbor_sampler,
                                                                       evaluate_idx_data_loader=test_idx_data_loader,
                                                                       evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                       evaluate_data=test_data,
                                                                       loss_func=loss_func,
                                                                       device=args.device,
                                                                       num_neighbors=args.num_neighbors,
                                                                       time_gap=args.time_gap,mode='test', loss_type = args.loss, full_data=full_data, collision_check=args.collision_check, dataset_name=args.dataset_name, logger=logger)
            # reload the model, so that the memory bank is reloaded
            early_stopping.load_checkpoint(model, map_location='cpu')
            model = convert_to_gpu(model, device=args.device)
            test_metrics_multi_negs = evaluate_model_link_prediction_multi_negs( \
                    model_name=args.model_name, \
                    model=model, \
                    neighbor_sampler=full_neighbor_sampler, \
                    evaluate_idx_data_loader=multi_negs_test_idx_data_loader, \
                    evaluate_neg_edge_sampler=test_neg_edge_sampler, \
                    evaluate_data=test_data, \
                    loss_func=loss_func, \
                    device=args.device, \
                    num_neighbors=args.num_neighbors, \
                    time_gap=args.time_gap, \
                    loss_type = args.loss, \
                    full_data=full_data, \
                    dataset_name=args.dataset_name, \
                    collision_check=args.collision_check, \
                    logger=logger)
        for metric_name in test_metrics.keys():
            logger.info(f'test {metric_name}, {test_metrics[metric_name]:.8f}')
        test_metrics.update(test_metrics_multi_negs)
        for metric_name in test_metrics.keys():
            logger.info(f'test {metric_name}, {test_metrics[metric_name]:.8f}')

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        test_metric_all_runs.append(test_metrics)

        result_json = {
            "test metrics": {metric_name: f'{test_metrics[metric_name]:.8f}' for metric_name in test_metrics},
        }
        result_json = json.dumps(result_json, indent=4)

        save_result_folder = f"./saved_results/{args.dataset_name}/{args.model_name}"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(
            save_result_folder, f"{args.save_model_name}.json")

        with open(save_result_path, 'w') as file:
            file.write(result_json)
        
        # write_results(save_model_folder, args.model_name, args.version, args.dataset_path, args.dataset_name, mrr_list, pos_rank_list, pos_scores, first_20_list, y_pred_first_5_list, last_nei_time_list)

    # store the average metrics at the log of the last run
    logger.info(f'metrics over {args.num_runs} runs:')

    for metric_name in test_metric_all_runs[0].keys():
        logger.info(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
        logger.info(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.8f} '
                    f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.8f}')
    print(log_file)
    sys.exit()
