"""
TPNet-style node property prediction for TGB node prediction datasets.

This mirrors the chronological replay in run_node_pred_v2.py, but augments the
input with node-wise temporal random-projection features from TPNet's
RandomProjectionModule.
"""

import sys

sys.path.insert(0, "/data/chenlibin/CRAFT_v2")
sys.path.insert(0, "/data/chenlibin/TGB2/src")

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tgb.nodeproppred.dataset import NodePropPredDataset
from tgb.nodeproppred.evaluate import Evaluator

from models.TPNet import TPNet, RandomProjectionModule


class IncrementalNodeState:
    """Tracks graph features + label history per node."""

    def __init__(self, num_nodes, num_classes, ema_alpha=0.3):
        self.num_nodes = num_nodes
        self.num_classes = num_classes
        self.ema_alpha = ema_alpha

        self.out_degree = np.zeros(num_nodes, dtype=np.int32)
        self.in_degree = np.zeros(num_nodes, dtype=np.int32)
        self.last_time_src = np.full(num_nodes, -1e9)
        self.last_time_dst = np.full(num_nodes, -1e9)
        self.n_neighbors = np.zeros(num_nodes, dtype=np.int32)
        self._nbr_sets = [set() for _ in range(num_nodes)]

        self.label_ema = np.zeros((num_nodes, num_classes), dtype=np.float32)
        self.label_count = np.zeros(num_nodes, dtype=np.int32)

    def update_edge(self, src, dst, t):
        self.out_degree[src] += 1
        self.in_degree[dst] += 1
        self.last_time_src[src] = t
        self.last_time_dst[dst] = t
        if dst not in self._nbr_sets[src]:
            self._nbr_sets[src].add(dst)
            self.n_neighbors[src] += 1
        if src not in self._nbr_sets[dst]:
            self._nbr_sets[dst].add(src)
            self.n_neighbors[dst] += 1

    def update_labels(self, node_ids, labels):
        for i, nid in enumerate(node_ids):
            nid = int(nid)
            if self.label_count[nid] == 0:
                self.label_ema[nid] = labels[i]
            else:
                self.label_ema[nid] = (1 - self.ema_alpha) * self.label_ema[nid] + self.ema_alpha * labels[i]
            self.label_count[nid] += 1

    def get_node_features(self, node, t):
        src_rec = t - self.last_time_src[node]
        dst_rec = t - self.last_time_dst[node]
        return np.array(
            [
                np.log1p(self.out_degree[node]),
                np.log1p(self.in_degree[node]),
                np.log1p(max(0, src_rec)) if src_rec < 1e8 else -1.0,
                np.log1p(max(0, dst_rec)) if dst_rec < 1e8 else -1.0,
                np.log1p(self.n_neighbors[node]),
                np.log1p(self.label_count[node]),
            ],
            dtype=np.float32,
        )

    def get_combined_features(self, node_ids, t):
        feats = np.zeros((len(node_ids), self.num_classes + 6), dtype=np.float32)
        for i, nid in enumerate(node_ids):
            nid = int(nid)
            feats[i, : self.num_classes] = self.label_ema[nid]
            feats[i, self.num_classes :] = self.get_node_features(nid, t)
        return feats


class TPNetNodePredictor(nn.Module):
    def __init__(self, tpnet_dim, num_classes, hidden_dim):
        super().__init__()
        input_dim = tpnet_dim + num_classes + 6
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


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
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1.")
    if args.hidden_dim < 1:
        raise ValueError("--hidden_dim must be >= 1.")


def normalize_soft_labels(labels):
    label_sums = labels.sum(axis=1, keepdims=True)
    label_sums = np.maximum(label_sums, 1e-8)
    return labels / label_sums


def update_edges_until(target_time, edge_idx, edge_limit, sources, destinations, timestamps, state, rp_module):
    while edge_idx < edge_limit and float(timestamps[edge_idx]) < target_time:
        batch_start = edge_idx
        batch_time = float(timestamps[batch_start])
        while edge_idx < edge_limit and float(timestamps[edge_idx]) == batch_time and float(timestamps[edge_idx]) < target_time:
            state.update_edge(int(sources[edge_idx]), int(destinations[edge_idx]), float(timestamps[edge_idx]))
            edge_idx += 1
        rp_module.update(
            src_node_ids=np.asarray(sources[batch_start:edge_idx], dtype=np.int64),
            dst_node_ids=np.asarray(destinations[batch_start:edge_idx], dtype=np.int64),
            node_interact_times=np.asarray(timestamps[batch_start:edge_idx], dtype=np.float64),
        )
    return edge_idx


def compute_tpnet_node_embeddings(rp_module, node_ids):
    # TPNet.py exposes node-wise temporal walk state through the random projection bank.
    # For node prediction we use the concatenation of A^(0)...A^(K) as the node embedding,
    # since there is no separate standalone node-embedding head defined for this setting.
    node_index = torch.as_tensor(node_ids, device=rp_module.random_projections[0].device, dtype=torch.long)
    with torch.no_grad():
        layer_embeddings = [rp_module.random_projections[layer][node_index] for layer in range(rp_module.num_layer + 1)]
        node_embeddings = torch.cat(layer_embeddings, dim=1)
        node_embeddings = torch.nan_to_num(node_embeddings, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
    return node_embeddings.detach().cpu().numpy().astype(np.float32)


def build_snapshot(feature_state, rp_module, labels_at_ts, num_classes, timestamp):
    node_ids = np.asarray(sorted(labels_at_ts.keys()), dtype=np.int64)
    labels = np.asarray([labels_at_ts[int(node_id)] for node_id in node_ids], dtype=np.float32)
    state_features = feature_state.get_combined_features(node_ids, timestamp)
    tpnet_embeddings = compute_tpnet_node_embeddings(rp_module, node_ids)
    features = np.concatenate(
        [
            tpnet_embeddings,
            state_features[:, :num_classes],
            state_features[:, num_classes:],
        ],
        axis=1,
    ).astype(np.float32)
    return {"node_ids": node_ids, "features": features, "labels": labels}


def collect_snapshots(split_timestamps, edge_idx, edge_limit, sources, destinations, timestamps, node_label_dict, state, rp_module, num_classes):
    snapshots = []
    for ts in split_timestamps:
        t_val = float(ts)
        edge_idx = update_edges_until(
            target_time=t_val,
            edge_idx=edge_idx,
            edge_limit=edge_limit,
            sources=sources,
            destinations=destinations,
            timestamps=timestamps,
            state=state,
            rp_module=rp_module,
        )

        labels_at_ts = node_label_dict[ts]
        if not labels_at_ts:
            continue

        snapshot = build_snapshot(state, rp_module, labels_at_ts, num_classes, t_val)
        snapshots.append(snapshot)
        state.update_labels(snapshot["node_ids"], snapshot["labels"])

    return snapshots, edge_idx


def normalize_feature_splits(train_features, snapshot_list, tpnet_dim, num_classes):
    struct_start = tpnet_dim + num_classes
    tpnet_mean = train_features[:, :tpnet_dim].mean(axis=0)
    tpnet_std = train_features[:, :tpnet_dim].std(axis=0) + 1e-8
    struct_mean = train_features[:, struct_start:].mean(axis=0)
    struct_std = train_features[:, struct_start:].std(axis=0) + 1e-8

    train_features[:, :tpnet_dim] = (train_features[:, :tpnet_dim] - tpnet_mean) / tpnet_std
    train_features[:, struct_start:] = (train_features[:, struct_start:] - struct_mean) / struct_std

    for snapshot in snapshot_list:
        snapshot["features"][:, :tpnet_dim] = (snapshot["features"][:, :tpnet_dim] - tpnet_mean) / tpnet_std
        snapshot["features"][:, struct_start:] = (snapshot["features"][:, struct_start:] - struct_mean) / struct_std


def evaluate_snapshots(model, snapshots, evaluator, device):
    if not snapshots:
        return 0.0

    metric_list = []
    model.eval()
    with torch.no_grad():
        for snapshot in snapshots:
            features = torch.from_numpy(snapshot["features"]).to(device=device, dtype=torch.float32)
            logits = model(features)
            pred_probs = F.softmax(logits, dim=-1).cpu().numpy()
            result = evaluator.eval(
                {
                    "y_true": snapshot["labels"],
                    "y_pred": pred_probs,
                    "eval_metric": ["ndcg"],
                }
            )
            metric_list.append(result["ndcg"])
    return float(np.mean(metric_list))


def train_and_evaluate(args):
    validate_args(args)
    seed_everything(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print(f"Loading {args.dataset}...")
    dataset = NodePropPredDataset(name=args.dataset, root="datasets")
    data = dataset.full_data
    sources = data["sources"]
    destinations = data["destinations"]
    timestamps = data["timestamps"]
    if "node_label_dict" in data:
        node_label_dict = data["node_label_dict"]
    elif hasattr(dataset, "label_dict"):
        node_label_dict = dataset.label_dict
    else:
        raise KeyError(f"{args.dataset}: neither full_data['node_label_dict'] nor dataset.label_dict available")

    ts_keys = sorted(node_label_dict.keys())
    if not ts_keys:
        raise ValueError(f"{args.dataset} has no node label snapshots.")

    first_nonempty = next((node_label_dict[ts] for ts in ts_keys if node_label_dict[ts]), None)
    if first_nonempty is None:
        raise ValueError(f"{args.dataset} has no non-empty node label snapshots.")

    first_node = next(iter(first_nonempty))
    num_classes = len(first_nonempty[first_node])
    max_edge_node = int(max(sources.max(), destinations.max()))
    max_label_node = max(int(max(labels_at_ts.keys())) for labels_at_ts in node_label_dict.values() if labels_at_ts)
    num_nodes = max(max_edge_node, max_label_node) + 1
    total_edges = len(sources)

    train_end = int(dataset.train_mask.sum())
    val_end = train_end + int(dataset.val_mask.sum())
    train_cutoff = float(timestamps[train_end - 1])
    val_cutoff = float(timestamps[val_end - 1]) if val_end > train_end else train_cutoff
    train_ts = [ts for ts in ts_keys if float(ts) <= train_cutoff]
    val_ts = [ts for ts in ts_keys if train_cutoff < float(ts) <= val_cutoff]
    test_ts = [ts for ts in ts_keys if float(ts) > val_cutoff]

    evaluator = Evaluator(name=args.dataset)
    if dataset.metric != "ndcg":
        raise ValueError(f"Expected ndcg metric, got {dataset.metric}.")

    print(f"Nodes: {num_nodes}, Edges: {total_edges}, Classes: {num_classes}")
    print(f"Train label ts: {len(train_ts)}, Val label ts: {len(val_ts)}, Test label ts: {len(test_ts)}")

    state = IncrementalNodeState(num_nodes=num_nodes, num_classes=num_classes, ema_alpha=args.ema_alpha)
    _ = TPNet
    rp_module = RandomProjectionModule(
        node_num=num_nodes,
        edge_num=total_edges,
        dim_factor=args.tpnet_dim_factor,
        num_layer=args.tpnet_num_layer,
        time_decay_weight=args.tpnet_time_decay,
        device=str(device),
        use_matrix=False,
        beginning_time=np.float64(np.min(timestamps)),
        not_scale=False,
        enforce_dim=-1,
    )
    rp_module = rp_module.to(device)
    tpnet_dim = (rp_module.num_layer + 1) * rp_module.dim

    edge_idx = 0
    train_snapshots, edge_idx = collect_snapshots(
        split_timestamps=train_ts,
        edge_idx=edge_idx,
        edge_limit=train_end,
        sources=sources,
        destinations=destinations,
        timestamps=timestamps,
        node_label_dict=node_label_dict,
        state=state,
        rp_module=rp_module,
        num_classes=num_classes,
    )
    edge_idx = update_edges_until(
        target_time=float("inf"),
        edge_idx=edge_idx,
        edge_limit=train_end,
        sources=sources,
        destinations=destinations,
        timestamps=timestamps,
        state=state,
        rp_module=rp_module,
    )
    if not train_snapshots:
        raise ValueError(f"{args.dataset} has no training label snapshots before the validation cutoff.")

    val_snapshots, edge_idx = collect_snapshots(
        split_timestamps=val_ts,
        edge_idx=edge_idx,
        edge_limit=val_end,
        sources=sources,
        destinations=destinations,
        timestamps=timestamps,
        node_label_dict=node_label_dict,
        state=state,
        rp_module=rp_module,
        num_classes=num_classes,
    )
    edge_idx = update_edges_until(
        target_time=float("inf"),
        edge_idx=edge_idx,
        edge_limit=val_end,
        sources=sources,
        destinations=destinations,
        timestamps=timestamps,
        state=state,
        rp_module=rp_module,
    )
    test_snapshots, edge_idx = collect_snapshots(
        split_timestamps=test_ts,
        edge_idx=edge_idx,
        edge_limit=total_edges,
        sources=sources,
        destinations=destinations,
        timestamps=timestamps,
        node_label_dict=node_label_dict,
        state=state,
        rp_module=rp_module,
        num_classes=num_classes,
    )

    train_features = np.concatenate([snapshot["features"] for snapshot in train_snapshots], axis=0)
    train_labels = np.concatenate([snapshot["labels"] for snapshot in train_snapshots], axis=0)

    all_eval_snapshots = val_snapshots + test_snapshots
    normalize_feature_splits(train_features, all_eval_snapshots, tpnet_dim=tpnet_dim, num_classes=num_classes)
    train_targets = normalize_soft_labels(train_labels)

    train_x = torch.from_numpy(train_features).to(device=device, dtype=torch.float32)
    train_y = torch.from_numpy(train_targets).to(device=device, dtype=torch.float32)

    model = TPNetNodePredictor(tpnet_dim=tpnet_dim, num_classes=num_classes, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    best_val_ndcg = -float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        logits = model(train_x)
        loss = F.kl_div(F.log_softmax(logits, dim=-1), train_y, reduction="batchmean")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        val_ndcg = evaluate_snapshots(model, val_snapshots, evaluator, device)
        if val_ndcg > best_val_ndcg:
            best_val_ndcg = val_ndcg
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        print(f"epoch {epoch} train_loss {loss.item():.6f} val_ndcg {val_ndcg:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    test_ndcg = evaluate_snapshots(model, test_snapshots, evaluator, device)
    print(f"Final Test NDCG: {test_ndcg:.6f}")

    os.makedirs("/data/chenlibin/TGB2/results", exist_ok=True)
    out_path = f"/data/chenlibin/TGB2/results/{args.dataset}_tpnet_seed{args.seed}.json"
    result = {
        "test_ndcg": float(test_ndcg),
        "val_ndcg": float(0.0 if best_val_ndcg == -float('inf') else best_val_ndcg),
        "config": {
            **vars(args),
            "tpnet_emb_dim": tpnet_dim,
            "num_classes": num_classes,
        },
    }
    with open(out_path, "w") as outfile:
        json.dump(result, outfile, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="tgbn-trade")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ema_alpha", type=float, default=0.8)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--tpnet_dim_factor", type=int, default=10)
    parser.add_argument("--tpnet_num_layer", type=int, default=2)
    parser.add_argument("--tpnet_time_decay", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    train_and_evaluate(parser.parse_args())
