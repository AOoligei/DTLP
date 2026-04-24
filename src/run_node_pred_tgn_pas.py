"""
TGN + PAS node property prediction for TGB node prediction benchmarks.

This script follows the official TGN node prediction replay loop while adding:
1. A pair-aware auxiliary loss via temporal link prediction on the same edge stream.
2. Incremental label EMA features concatenated to TGN node embeddings.
3. Optional orthogonality regularization between node and pair projections.
"""

import argparse
import json
import os
import random
import sys
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
TGB_BASELINES_ROOT = os.path.join(REPO_ROOT, "tgb_baselines")

for path in (SCRIPT_DIR, REPO_ROOT, TGB_BASELINES_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from modules.decoder import LinkPredictor
from modules.emb_module import GraphAttentionEmbedding
from tgb.nodeproppred.dataset_pyg import PyGNodePropPredDataset
from tgb.nodeproppred.evaluate import Evaluator


SUPPORTED_DATASETS = {"tgbn-trade", "tgbn-token"}
DEFAULT_EPOCHS = 50
DEFAULT_LR = 1e-4
DEFAULT_EMA_ALPHA = 0.8
DEFAULT_HIDDEN_DIM = 100
DEFAULT_BATCH_SIZE = 200
DEFAULT_NUM_NEIGHBORS = 10
DEFAULT_MEMORY_DIM = 100
DEFAULT_TIME_DIM = 100
DEFAULT_EMBEDDING_DIM = 100
DEFAULT_LAMBDA_PAIR = 1.0
DEFAULT_LAMBDA_ORTH = 0.0
NUM_PAIR_NEGATIVES = 1
EPSILON = 1e-8
EMBEDDING_CLAMP_VALUE = 1e4
RESULTS_DIRNAME = "results"


class IncrementalNodeState:
    """Tracks structural history and label EMA features for each node."""

    def __init__(self, num_nodes, num_classes, ema_alpha=DEFAULT_EMA_ALPHA):
        self.num_nodes = num_nodes
        self.num_classes = num_classes
        self.ema_alpha = ema_alpha
        self.reset()

    def reset(self):
        self.out_degree = np.zeros(self.num_nodes, dtype=np.int32)
        self.in_degree = np.zeros(self.num_nodes, dtype=np.int32)
        self.last_time_src = np.full(self.num_nodes, -1e9, dtype=np.float64)
        self.last_time_dst = np.full(self.num_nodes, -1e9, dtype=np.float64)
        self.n_neighbors = np.zeros(self.num_nodes, dtype=np.int32)
        self._nbr_sets = [set() for _ in range(self.num_nodes)]
        self.label_ema = np.zeros((self.num_nodes, self.num_classes), dtype=np.float32)
        self.label_count = np.zeros(self.num_nodes, dtype=np.int32)

    def update_edges(self, src_node_ids, dst_node_ids, node_interact_times):
        src_node_ids = np.asarray(src_node_ids, dtype=np.int64)
        dst_node_ids = np.asarray(dst_node_ids, dtype=np.int64)
        node_interact_times = np.asarray(node_interact_times, dtype=np.float64)

        for src, dst, ts in zip(src_node_ids, dst_node_ids, node_interact_times):
            src = int(src)
            dst = int(dst)
            ts = float(ts)
            self.out_degree[src] += 1
            self.in_degree[dst] += 1
            self.last_time_src[src] = ts
            self.last_time_dst[dst] = ts

            if dst not in self._nbr_sets[src]:
                self._nbr_sets[src].add(dst)
                self.n_neighbors[src] += 1
            if src not in self._nbr_sets[dst]:
                self._nbr_sets[dst].add(src)
                self.n_neighbors[dst] += 1

    def update_labels(self, node_ids, labels):
        node_ids = np.asarray(node_ids, dtype=np.int64)
        labels = np.asarray(labels, dtype=np.float32)
        for idx, node_id in enumerate(node_ids):
            node_id = int(node_id)
            if self.label_count[node_id] == 0:
                self.label_ema[node_id] = labels[idx]
            else:
                self.label_ema[node_id] = (
                    (1.0 - self.ema_alpha) * self.label_ema[node_id]
                    + self.ema_alpha * labels[idx]
                )
            self.label_count[node_id] += 1

    def get_node_features(self, node_id, current_time):
        src_recency = current_time - self.last_time_src[node_id]
        dst_recency = current_time - self.last_time_dst[node_id]
        return np.array(
            [
                np.log1p(self.out_degree[node_id]),
                np.log1p(self.in_degree[node_id]),
                np.log1p(max(0.0, src_recency)) if src_recency < 1e8 else -1.0,
                np.log1p(max(0.0, dst_recency)) if dst_recency < 1e8 else -1.0,
                np.log1p(self.n_neighbors[node_id]),
                np.log1p(self.label_count[node_id]),
            ],
            dtype=np.float32,
        )

    def get_combined_features(self, node_ids, current_time):
        node_ids = np.asarray(node_ids, dtype=np.int64)
        features = np.zeros((len(node_ids), self.num_classes + 6), dtype=np.float32)
        for idx, node_id in enumerate(node_ids):
            node_id = int(node_id)
            features[idx, : self.num_classes] = self.label_ema[node_id]
            features[idx, self.num_classes :] = self.get_node_features(node_id, current_time)
        return features


class PairAwareTGNModel(nn.Module):
    """TGN readout heads for node prediction + pair-aware supervision."""

    def __init__(self, embedding_dim, aux_feature_dim, num_classes, hidden_dim):
        super().__init__()
        self.node_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.pair_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.node_mlp = nn.Sequential(
            nn.Linear(embedding_dim + aux_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.link_pred = LinkPredictor(embedding_dim)
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward_node(self, node_embeddings, aux_features):
        node_embeddings = self.node_proj(node_embeddings)
        return self.node_mlp(torch.cat([node_embeddings, aux_features], dim=-1))

    def forward_pair(self, src_embeddings, dst_embeddings):
        src_pair = self.pair_proj(src_embeddings)
        dst_pair = self.pair_proj(dst_embeddings)
        return self.link_pred(src_pair, dst_pair).view(-1)

    def orthogonality_loss(self):
        node_weight = F.normalize(self.node_proj.weight, dim=1, eps=EPSILON)
        pair_weight = F.normalize(self.pair_proj.weight, dim=1, eps=EPSILON)
        return (node_weight @ pair_weight.t()).pow(2).mean()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def validate_args(args):
    if args.dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"--dataset must be one of {sorted(SUPPORTED_DATASETS)}.")
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1.")
    if args.hidden_dim < 1:
        raise ValueError("--hidden_dim must be >= 1.")
    if args.lr <= 0:
        raise ValueError("--lr must be > 0.")
    if not 0.0 <= args.ema_alpha <= 1.0:
        raise ValueError("--ema_alpha must be in [0, 1].")
    if args.lambda_pair < 0:
        raise ValueError("--lambda_pair must be >= 0.")
    if args.lambda_orth < 0:
        raise ValueError("--lambda_orth must be >= 0.")


def normalize_soft_labels(labels):
    label_sums = labels.sum(dim=-1, keepdim=True).clamp_min(EPSILON)
    return labels / label_sums


def peek_next_label_time(dataset, previous_label_t=None):
    label_t = dataset.get_label_time()
    if label_t is None:
        return float("inf")
    label_t = float(label_t)
    if previous_label_t is not None and label_t <= previous_label_t:
        return float("inf")
    return label_t


def stabilize_tensor(tensor):
    return torch.nan_to_num(
        tensor,
        nan=0.0,
        posinf=EMBEDDING_CLAMP_VALUE,
        neginf=-EMBEDDING_CLAMP_VALUE,
    ).clamp(-EMBEDDING_CLAMP_VALUE, EMBEDDING_CLAMP_VALUE)


def count_parameters(*modules):
    return sum(
        parameter.numel()
        for module in modules
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def sample_negative_destinations(candidate_dst_nodes, positive_dst):
    neg_indices = torch.randint(
        low=0,
        high=candidate_dst_nodes.numel(),
        size=(positive_dst.size(0) * NUM_PAIR_NEGATIVES,),
        device=positive_dst.device,
    )
    negative_dst = candidate_dst_nodes[neg_indices]
    if candidate_dst_nodes.numel() > 1:
        collision_mask = negative_dst == positive_dst.repeat_interleave(NUM_PAIR_NEGATIVES)
        while collision_mask.any():
            replacement = torch.randint(
                low=0,
                high=candidate_dst_nodes.numel(),
                size=(int(collision_mask.sum().item()),),
                device=positive_dst.device,
            )
            negative_dst[collision_mask] = candidate_dst_nodes[replacement]
            collision_mask = negative_dst == positive_dst.repeat_interleave(NUM_PAIR_NEGATIVES)
    return negative_dst


def encode_query_nodes(query_nodes, memory, gnn, neighbor_loader, full_data, assoc):
    node_ids, edge_index, edge_ids = neighbor_loader(query_nodes)
    assoc[node_ids] = torch.arange(node_ids.size(0), device=query_nodes.device)

    node_embeddings, last_update = memory(node_ids)
    if edge_ids.numel() > 0:
        node_embeddings = gnn(
            node_embeddings,
            last_update,
            edge_index,
            full_data.t[edge_ids].to(query_nodes.device),
            full_data.msg[edge_ids].to(query_nodes.device),
        )
    return stabilize_tensor(node_embeddings), node_ids


def update_temporal_state(src, dst, t, msg, memory, neighbor_loader, feature_state):
    if src.numel() == 0:
        return
    memory.update_state(src, dst, t, msg)
    neighbor_loader.insert(src, dst)
    feature_state.update_edges(
        src.detach().cpu().numpy(),
        dst.detach().cpu().numpy(),
        t.detach().cpu().numpy(),
    )


def compute_pair_loss_and_update(
    src,
    dst,
    t,
    msg,
    model,
    memory,
    gnn,
    neighbor_loader,
    full_data,
    assoc,
    candidate_dst_nodes,
    feature_state,
    device,
):
    if src.numel() == 0:
        update_temporal_state(src, dst, t, msg, memory, neighbor_loader, feature_state)
        return torch.zeros((), device=device)

    negative_dst = sample_negative_destinations(candidate_dst_nodes, dst)
    src_repeated = src.repeat_interleave(NUM_PAIR_NEGATIVES)
    query_nodes = torch.cat([src, dst, negative_dst]).unique()
    node_embeddings, _ = encode_query_nodes(
        query_nodes, memory, gnn, neighbor_loader, full_data, assoc
    )

    src_embeddings = node_embeddings[assoc[src]]
    pos_dst_embeddings = node_embeddings[assoc[dst]]
    neg_dst_embeddings = node_embeddings[assoc[negative_dst]]
    neg_src_embeddings = node_embeddings[assoc[src_repeated]]

    pos_prob = model.forward_pair(src_embeddings, pos_dst_embeddings).clamp(EPSILON, 1.0 - EPSILON)
    neg_prob = model.forward_pair(neg_src_embeddings, neg_dst_embeddings).clamp(EPSILON, 1.0 - EPSILON)

    pos_loss = F.binary_cross_entropy(pos_prob, torch.ones_like(pos_prob))
    neg_loss = F.binary_cross_entropy(neg_prob, torch.zeros_like(neg_prob))
    pair_loss = 0.5 * (pos_loss + neg_loss)

    update_temporal_state(src, dst, t, msg, memory, neighbor_loader, feature_state)
    return pair_loss


def compute_node_logits(
    label_srcs,
    label_timestamp,
    model,
    memory,
    gnn,
    neighbor_loader,
    full_data,
    assoc,
    feature_state,
    device,
):
    node_embeddings, _ = encode_query_nodes(
        label_srcs, memory, gnn, neighbor_loader, full_data, assoc
    )
    node_embeddings = node_embeddings[assoc[label_srcs]]
    aux_features = feature_state.get_combined_features(
        label_srcs.detach().cpu().numpy(), label_timestamp
    )
    aux_features = torch.from_numpy(aux_features).to(device=device, dtype=torch.float32)
    return model.forward_node(node_embeddings, aux_features)


def compute_ndcg(logits, labels, evaluator):
    probabilities = F.softmax(logits, dim=-1).detach().cpu().numpy()
    true_labels = labels.detach().cpu().numpy()
    return float(
        evaluator.eval(
            {
                "y_true": true_labels,
                "y_pred": probabilities,
                "eval_metric": ["ndcg"],
            }
        )["ndcg"]
    )


def train_one_epoch(
    args,
    dataset,
    train_loader,
    model,
    memory,
    gnn,
    neighbor_loader,
    optimizer,
    full_data,
    assoc,
    feature_state,
    candidate_dst_nodes,
    evaluator,
    device,
):
    memory.train()
    gnn.train()
    model.train()

    memory.reset_state()
    neighbor_loader.reset_state()
    feature_state.reset()

    label_t = peek_next_label_time(dataset)
    total_node_loss = 0.0
    total_pair_loss = 0.0
    total_orth_loss = 0.0
    total_score = 0.0
    num_label_ts = 0
    num_optim_steps = 0

    def run_optimizer_step(total_loss, pair_loss_value=0.0, orth_loss_value=0.0):
        nonlocal total_pair_loss, total_orth_loss, num_optim_steps

        optimizer.zero_grad(set_to_none=True)
        if total_loss.requires_grad:
            total_loss.backward()
        optimizer.step()
        memory.detach()

        total_pair_loss += float(pair_loss_value)
        total_orth_loss += float(orth_loss_value)
        num_optim_steps += 1

    for batch in train_loader:
        batch = batch.to(device)
        src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        while src.numel() > 0 and float(t[-1].item()) > label_t:
            current_label_t = label_t
            label_tuple = dataset.get_node_label(int(t[-1].item()))
            if label_tuple is not None:
                label_ts, label_srcs, labels = label_tuple
                label_srcs = label_srcs.to(device)
                labels = labels.to(device)
                label_timestamp = (
                    float(label_ts[0].item()) if label_ts.numel() > 0 else float(t[-1].item())
                )

                next_label_t = peek_next_label_time(dataset, current_label_t)
                previous_day_mask = t < next_label_t

                pair_loss = compute_pair_loss_and_update(
                    src=src[previous_day_mask],
                    dst=dst[previous_day_mask],
                    t=t[previous_day_mask],
                    msg=msg[previous_day_mask],
                    model=model,
                    memory=memory,
                    gnn=gnn,
                    neighbor_loader=neighbor_loader,
                    full_data=full_data,
                    assoc=assoc,
                    candidate_dst_nodes=candidate_dst_nodes,
                    feature_state=feature_state,
                    device=device,
                )
                pair_loss_value = float(pair_loss.detach().item())

                src, dst, t, msg = (
                    src[~previous_day_mask],
                    dst[~previous_day_mask],
                    t[~previous_day_mask],
                    msg[~previous_day_mask],
                )

                logits = compute_node_logits(
                    label_srcs=label_srcs,
                    label_timestamp=label_timestamp,
                    model=model,
                    memory=memory,
                    gnn=gnn,
                    neighbor_loader=neighbor_loader,
                    full_data=full_data,
                    assoc=assoc,
                    feature_state=feature_state,
                    device=device,
                )
                node_loss = F.kl_div(
                    F.log_softmax(logits, dim=-1),
                    normalize_soft_labels(labels),
                    reduction="batchmean",
                )
                orth_loss = (
                    model.orthogonality_loss()
                    if args.lambda_orth > 0
                    else torch.zeros((), device=device)
                )
                total_loss = (
                    node_loss
                    + args.lambda_pair * pair_loss
                    + args.lambda_orth * orth_loss
                )

                run_optimizer_step(
                    total_loss,
                    pair_loss_value=pair_loss_value,
                    orth_loss_value=float(orth_loss.detach().item()),
                )

                total_node_loss += float(node_loss.detach().item())
                total_score += compute_ndcg(logits, labels, evaluator)
                num_label_ts += 1

                feature_state.update_labels(
                    label_srcs.detach().cpu().numpy(),
                    labels.detach().cpu().numpy(),
                )
                label_t = next_label_t
            else:
                label_t = float("inf")
                break

        if src.numel() > 0:
            pair_loss = compute_pair_loss_and_update(
                src=src,
                dst=dst,
                t=t,
                msg=msg,
                model=model,
                memory=memory,
                gnn=gnn,
                neighbor_loader=neighbor_loader,
                full_data=full_data,
                assoc=assoc,
                candidate_dst_nodes=candidate_dst_nodes,
                feature_state=feature_state,
                device=device,
            )
            run_optimizer_step(
                args.lambda_pair * pair_loss,
                pair_loss_value=float(pair_loss.detach().item()),
            )

    if num_label_ts == 0 and num_optim_steps == 0:
        return {"node_loss": 0.0, "pair_loss": 0.0, "orth_loss": 0.0, "ndcg": 0.0}

    return {
        "node_loss": total_node_loss / num_label_ts if num_label_ts > 0 else 0.0,
        "pair_loss": total_pair_loss / num_optim_steps if num_optim_steps > 0 else 0.0,
        "orth_loss": total_orth_loss / num_optim_steps if num_optim_steps > 0 else 0.0,
        "ndcg": total_score / num_label_ts if num_label_ts > 0 else 0.0,
    }


@torch.no_grad()
def evaluate_split(
    dataset,
    loader,
    model,
    memory,
    gnn,
    neighbor_loader,
    full_data,
    assoc,
    feature_state,
    evaluator,
    device,
):
    memory.eval()
    gnn.eval()
    model.eval()

    label_t = peek_next_label_time(dataset)
    total_score = 0.0
    num_label_ts = 0

    for batch in loader:
        batch = batch.to(device)
        src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        while src.numel() > 0 and float(t[-1].item()) > label_t:
            current_label_t = label_t
            label_tuple = dataset.get_node_label(int(t[-1].item()))
            if label_tuple is not None:
                label_ts, label_srcs, labels = label_tuple
                label_srcs = label_srcs.to(device)
                labels = labels.to(device)
                label_timestamp = (
                    float(label_ts[0].item()) if label_ts.numel() > 0 else float(t[-1].item())
                )

                next_label_t = peek_next_label_time(dataset, current_label_t)
                previous_day_mask = t < next_label_t
                update_temporal_state(
                    src[previous_day_mask],
                    dst[previous_day_mask],
                    t[previous_day_mask],
                    msg[previous_day_mask],
                    memory,
                    neighbor_loader,
                    feature_state,
                )

                src, dst, t, msg = (
                    src[~previous_day_mask],
                    dst[~previous_day_mask],
                    t[~previous_day_mask],
                    msg[~previous_day_mask],
                )

                logits = compute_node_logits(
                    label_srcs=label_srcs,
                    label_timestamp=label_timestamp,
                    model=model,
                    memory=memory,
                    gnn=gnn,
                    neighbor_loader=neighbor_loader,
                    full_data=full_data,
                    assoc=assoc,
                    feature_state=feature_state,
                    device=device,
                )
                total_score += compute_ndcg(logits, labels, evaluator)
                num_label_ts += 1

                feature_state.update_labels(
                    label_srcs.detach().cpu().numpy(),
                    labels.detach().cpu().numpy(),
                )
                label_t = next_label_t
            else:
                label_t = float("inf")
                break

        update_temporal_state(src, dst, t, msg, memory, neighbor_loader, feature_state)

    if num_label_ts == 0:
        return {"ndcg": 0.0}
    return {"ndcg": total_score / num_label_ts}


def train_and_evaluate(args):
    validate_args(args)
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {args.dataset}...")
    dataset = PyGNodePropPredDataset(
        name=args.dataset,
        root="datasets",
    )
    if dataset.eval_metric != "ndcg":
        raise ValueError(f"Expected ndcg metric for {args.dataset}, got {dataset.eval_metric}.")

    evaluator = Evaluator(name=args.dataset)
    num_classes = dataset.num_classes
    full_data = dataset.get_TemporalData().to(device)
    num_edges = int(full_data.src.size(0))

    train_data = full_data[dataset.train_mask]
    val_data = full_data[dataset.val_mask]
    test_data = full_data[dataset.test_mask]

    train_loader = TemporalDataLoader(train_data, batch_size=DEFAULT_BATCH_SIZE)
    val_loader = TemporalDataLoader(val_data, batch_size=DEFAULT_BATCH_SIZE)
    test_loader = TemporalDataLoader(test_data, batch_size=DEFAULT_BATCH_SIZE)

    neighbor_loader = LastNeighborLoader(
        full_data.num_nodes, size=DEFAULT_NUM_NEIGHBORS, device=device
    )

    memory = TGNMemory(
        full_data.num_nodes,
        full_data.msg.size(-1),
        DEFAULT_MEMORY_DIM,
        DEFAULT_TIME_DIM,
        message_module=IdentityMessage(
            full_data.msg.size(-1), DEFAULT_MEMORY_DIM, DEFAULT_TIME_DIM
        ),
        aggregator_module=LastAggregator(),
    ).to(device)

    gnn = GraphAttentionEmbedding(
        in_channels=DEFAULT_MEMORY_DIM,
        out_channels=DEFAULT_EMBEDDING_DIM,
        msg_dim=full_data.msg.size(-1),
        time_enc=memory.time_enc,
    ).to(device).float()

    model = PairAwareTGNModel(
        embedding_dim=DEFAULT_EMBEDDING_DIM,
        aux_feature_dim=num_classes + 6,
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
    ).to(device)

    optimizer = torch.optim.Adam(
        chain(memory.parameters(), gnn.parameters(), model.parameters()),
        lr=args.lr,
    )

    assoc = torch.empty(full_data.num_nodes, dtype=torch.long, device=device)
    candidate_dst_nodes = torch.unique(full_data.dst)
    feature_state = IncrementalNodeState(
        num_nodes=full_data.num_nodes,
        num_classes=num_classes,
        ema_alpha=args.ema_alpha,
    )

    n_params = count_parameters(memory, gnn, model)
    print(
        f"Nodes: {full_data.num_nodes}, Edges: {num_edges}, "
        f"Classes: {num_classes}, Params: {n_params}"
    )

    best_val_ndcg = -float("inf")
    best_test_ndcg = 0.0

    for epoch in range(1, args.epochs + 1):
        dataset.reset_label_time()

        train_metrics = train_one_epoch(
            args=args,
            dataset=dataset,
            train_loader=train_loader,
            model=model,
            memory=memory,
            gnn=gnn,
            neighbor_loader=neighbor_loader,
            optimizer=optimizer,
            full_data=full_data,
            assoc=assoc,
            feature_state=feature_state,
            candidate_dst_nodes=candidate_dst_nodes,
            evaluator=evaluator,
            device=device,
        )
        val_metrics = evaluate_split(
            dataset=dataset,
            loader=val_loader,
            model=model,
            memory=memory,
            gnn=gnn,
            neighbor_loader=neighbor_loader,
            full_data=full_data,
            assoc=assoc,
            feature_state=feature_state,
            evaluator=evaluator,
            device=device,
        )
        test_metrics = evaluate_split(
            dataset=dataset,
            loader=test_loader,
            model=model,
            memory=memory,
            gnn=gnn,
            neighbor_loader=neighbor_loader,
            full_data=full_data,
            assoc=assoc,
            feature_state=feature_state,
            evaluator=evaluator,
            device=device,
        )

        if val_metrics["ndcg"] > best_val_ndcg:
            best_val_ndcg = val_metrics["ndcg"]
            best_test_ndcg = test_metrics["ndcg"]

        print(
            f"epoch {epoch:02d} "
            f"train_node_loss {train_metrics['node_loss']:.6f} "
            f"train_pair_loss {train_metrics['pair_loss']:.6f} "
            f"train_orth_loss {train_metrics['orth_loss']:.6f} "
            f"train_ndcg {train_metrics['ndcg']:.6f} "
            f"val_ndcg {val_metrics['ndcg']:.6f} "
            f"test_ndcg {test_metrics['ndcg']:.6f}"
        )

    result = {
        "dataset": args.dataset,
        "model": "TGN-Node-PAS",
        "config": {
            **vars(args),
            "batch_size": DEFAULT_BATCH_SIZE,
            "num_neighbors": DEFAULT_NUM_NEIGHBORS,
            "memory_dim": DEFAULT_MEMORY_DIM,
            "time_dim": DEFAULT_TIME_DIM,
            "embedding_dim": DEFAULT_EMBEDDING_DIM,
            "num_pair_negatives": NUM_PAIR_NEGATIVES,
            "num_classes": num_classes,
        },
        "ndcg": float(best_test_ndcg),
        "seed": args.seed,
        "n_params": int(n_params),
        "ema_alpha": float(args.ema_alpha),
        "lambda_pair": float(args.lambda_pair),
    }

    results_dir = os.path.join(REPO_ROOT, RESULTS_DIRNAME)
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, f"{args.dataset}_tgn_pas_seed{args.seed}.json")
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)

    print(f"Best validation NDCG: {best_val_ndcg:.6f}")
    print(f"Saved results to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="TGN + PAS node property prediction on TGB datasets.")
    parser.add_argument("--dataset", type=str, default="tgbn-trade", choices=sorted(SUPPORTED_DATASETS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ema_alpha", type=float, default=DEFAULT_EMA_ALPHA)
    parser.add_argument("--hidden_dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--lambda_pair", type=float, default=DEFAULT_LAMBDA_PAIR)
    parser.add_argument("--lambda_orth", type=float, default=DEFAULT_LAMBDA_ORTH)
    return parser.parse_args()


if __name__ == "__main__":
    train_and_evaluate(parse_args())
