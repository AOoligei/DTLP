"""
DTLP-Node v2: Node property prediction with label history features.

Key insight from NAVIS: the previous label distribution is the strongest predictor.
We combine:
  1. Historical label distribution (exponential moving average)
  2. Graph structural features (degree, recency, etc.)
  3. LambdaNDCG-style ranking loss (or softmax cross-entropy as proxy)

This is closer to a proper adaptation for the node prediction task.
"""

import sys
sys.path.insert(0, "/data/chenlibin/CRAFT_v2")
sys.path.insert(0, "/data/chenlibin/TGB2/src")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tgb.nodeproppred.dataset import NodePropPredDataset
from tgb.nodeproppred.evaluate import Evaluator
from sklearn.metrics import ndcg_score
import time
import json
import os
import argparse
from collections import defaultdict


def get_safe_result_path(base_dir, stem):
    """Avoid overwriting prior result files from earlier runs."""
    candidate = os.path.join(base_dir, f"{stem}.json")
    if not os.path.exists(candidate):
        return candidate
    suffix = 2
    while True:
        candidate = os.path.join(base_dir, f"{stem}_run{suffix}.json")
        if not os.path.exists(candidate):
            return candidate
        suffix += 1


def validate_args(args):
    """Fail fast on settings that otherwise crash later or produce NaNs."""
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1.")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1.")


class NodePredictorV2(nn.Module):
    """
    Node predictor that uses:
    - Historical label EMA (num_classes dim)
    - Graph structural features (6 dim)
    → MLP → num_classes logits
    """
    def __init__(self, num_classes, struct_feat_dim=6, hidden_dim=128, dropout=0.3):
        super().__init__()
        input_dim = num_classes + struct_feat_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class IncrementalNodeState:
    """Tracks graph features + label history per node."""
    def __init__(self, num_nodes, num_classes, ema_alpha=0.3):
        self.num_nodes = num_nodes
        self.num_classes = num_classes
        self.ema_alpha = ema_alpha

        # Graph features
        self.out_degree = np.zeros(num_nodes, dtype=np.int32)
        self.in_degree = np.zeros(num_nodes, dtype=np.int32)
        self.last_time_src = np.full(num_nodes, -1e9)
        self.last_time_dst = np.full(num_nodes, -1e9)
        self.n_neighbors = np.zeros(num_nodes, dtype=np.int32)
        self._nbr_sets = [set() for _ in range(num_nodes)]

        # Label history: exponential moving average of past labels
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
        """Update label EMA for nodes. labels: (n_nodes, num_classes)."""
        for i, nid in enumerate(node_ids):
            nid = int(nid)
            if self.label_count[nid] == 0:
                self.label_ema[nid] = labels[i]
            else:
                self.label_ema[nid] = (1 - self.ema_alpha) * self.label_ema[nid] + \
                                       self.ema_alpha * labels[i]
            self.label_count[nid] += 1

    def get_node_features(self, node, t):
        """6-dim structural features."""
        src_rec = t - self.last_time_src[node]
        dst_rec = t - self.last_time_dst[node]
        return np.array([
            np.log1p(self.out_degree[node]),
            np.log1p(self.in_degree[node]),
            np.log1p(max(0, src_rec)) if src_rec < 1e8 else -1.0,
            np.log1p(max(0, dst_rec)) if dst_rec < 1e8 else -1.0,
            np.log1p(self.n_neighbors[node]),
            np.log1p(self.label_count[node]),
        ], dtype=np.float32)

    def get_combined_features(self, node_ids, t):
        """Returns (n, num_classes + 6) feature matrix."""
        n = len(node_ids)
        feats = np.zeros((n, self.num_classes + 6), dtype=np.float32)
        for i, nid in enumerate(node_ids):
            nid = int(nid)
            feats[i, :self.num_classes] = self.label_ema[nid]
            feats[i, self.num_classes:] = self.get_node_features(nid, t)
        return feats


def train_and_evaluate(args):
    validate_args(args)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # 1. Load
    print(f"Loading {args.dataset}...")
    dataset = NodePropPredDataset(name=args.dataset, root="datasets")
    data = dataset.full_data
    sources = data['sources']
    destinations = data['destinations']
    timestamps = data['timestamps']
    if 'node_label_dict' in data:
        node_label_dict = data['node_label_dict']
    elif hasattr(dataset, 'label_dict'):
        node_label_dict = dataset.label_dict
    else:
        raise KeyError(f"{args.dataset}: neither full_data['node_label_dict'] nor dataset.label_dict available")

    num_nodes = int(max(sources.max(), destinations.max())) + 1
    total = len(sources)
    eval_metric = dataset.metric
    evaluator = Evaluator(name=args.dataset)

    ts_keys = sorted(node_label_dict.keys())
    if not ts_keys:
        raise ValueError(f"{args.dataset} has no node labels to train or evaluate on.")
    first_labels = next((node_label_dict[ts] for ts in ts_keys if node_label_dict[ts]), None)
    if first_labels is None:
        raise ValueError(f"{args.dataset} has no non-empty node label snapshots.")
    first_node = next(iter(first_labels))
    num_classes = len(first_labels[first_node])

    train_end = int(dataset.train_mask.sum())
    val_end = train_end + int(dataset.val_mask.sum())
    if train_end < 1:
        raise ValueError(f"{args.dataset} has no training edges.")

    print(f"  Nodes: {num_nodes}, Edges: {total}, Classes: {num_classes}")
    print(f"  Train: {train_end}, Val: {val_end-train_end}, Test: {total-val_end}")

    # 2. Build state
    state = IncrementalNodeState(num_nodes, num_classes, ema_alpha=args.ema_alpha)

    # Process training edges and labels
    print(f"\nProcessing training edges and labels...")
    t0 = time.time()

    train_ts = [ts for ts in ts_keys if ts <= timestamps[train_end - 1]]
    val_ts = [ts for ts in ts_keys if timestamps[train_end - 1] < ts <= timestamps[val_end - 1]]
    test_ts = [ts for ts in ts_keys if ts > timestamps[val_end - 1]]

    # Process edges up to train_end
    edge_idx = 0
    label_ts_idx = 0
    train_feats_list = []
    train_labels_list = []

    # Process chronologically: edges then labels at each timestamp
    for ts in train_ts:
        t_val = float(ts)
        # Process edges up to this label timestamp
        while edge_idx < train_end and float(timestamps[edge_idx]) < t_val:
            state.update_edge(int(sources[edge_idx]), int(destinations[edge_idx]),
                            float(timestamps[edge_idx]))
            edge_idx += 1

        # Get features BEFORE updating labels (predict from past)
        labels_at_ts = node_label_dict[ts]
        if not labels_at_ts:
            continue
        node_ids = list(labels_at_ts.keys())
        labels = np.array([labels_at_ts[n] for n in node_ids], dtype=np.float32)

        # Get combined features (label_ema + structural)
        feats = state.get_combined_features(node_ids, t_val)
        train_feats_list.append(feats)
        train_labels_list.append(labels)

        # NOW update label EMA with ground truth
        state.update_labels(node_ids, labels)

    # Process remaining training edges
    while edge_idx < train_end:
        state.update_edge(int(sources[edge_idx]), int(destinations[edge_idx]),
                        float(timestamps[edge_idx]))
        edge_idx += 1

    if not train_feats_list:
        raise ValueError(f"{args.dataset} has no training label snapshots before the validation cutoff.")
    train_feats = np.concatenate(train_feats_list, axis=0)
    train_labels = np.concatenate(train_labels_list, axis=0)
    n_train = len(train_feats)
    print(f"  Training samples: {n_train} ({time.time()-t0:.1f}s)")

    # Normalize structural features only (label EMA already in [0,1])
    struct_mean = train_feats[:, num_classes:].mean(0)
    struct_std = train_feats[:, num_classes:].std(0) + 1e-8
    train_feats[:, num_classes:] = (train_feats[:, num_classes:] - struct_mean) / struct_std

    # Normalize labels
    label_sums = train_labels.sum(axis=1, keepdims=True)
    label_sums = np.maximum(label_sums, 1e-8)
    train_labels_norm = train_labels / label_sums

    # 3. Model
    model = NodePredictorV2(num_classes, struct_feat_dim=6,
                            hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    t_feats = torch.FloatTensor(train_feats).to(device)
    t_labels = torch.FloatTensor(train_labels_norm).to(device)

    # 4. Train with KL divergence (TODO: LambdaNDCG for better NDCG optimization)
    checkpoints = {}
    best_loss = float('inf')
    best_epoch = 1
    patience_counter = 0
    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        total_loss = 0
        n_batches = 0

        for i in range(0, n_train, args.batch_size):
            idx = perm[i:i + args.batch_size]
            logits = model(t_feats[idx])
            log_probs = F.log_softmax(logits, dim=-1)
            targets = t_labels[idx]
            loss = F.kl_div(log_probs, targets, reduction='batchmean')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        checkpoints[epoch + 1] = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f} (best: ep{best_epoch} {best_loss:.4f})")
        if patience_counter >= 15 and epoch >= 10:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    if best_epoch in checkpoints:
        model.load_state_dict({k: v.to(device) for k, v in checkpoints[best_epoch].items()})
        print(f"  Restored best model from epoch {best_epoch} (loss={best_loss:.4f})")

    # 5. Evaluate
    print(f"\n{'='*60}")
    print("Evaluating...")
    model.eval()

    # Process val edges and labels
    for ts in val_ts:
        t_val = float(ts)
        while edge_idx < val_end and float(timestamps[edge_idx]) < t_val:
            state.update_edge(int(sources[edge_idx]), int(destinations[edge_idx]),
                            float(timestamps[edge_idx]))
            edge_idx += 1
        labels_at_ts = node_label_dict[ts]
        if not labels_at_ts:
            continue
        node_ids = list(labels_at_ts.keys())
        labels = np.array([labels_at_ts[n] for n in node_ids], dtype=np.float32)
        state.update_labels(node_ids, labels)

    while edge_idx < val_end:
        state.update_edge(int(sources[edge_idx]), int(destinations[edge_idx]),
                        float(timestamps[edge_idx]))
        edge_idx += 1

    # Test
    metric_list = []
    for ts in test_ts:
        t_val = float(ts)
        while edge_idx < total and float(timestamps[edge_idx]) < t_val:
            state.update_edge(int(sources[edge_idx]), int(destinations[edge_idx]),
                            float(timestamps[edge_idx]))
            edge_idx += 1

        labels_at_ts = node_label_dict[ts]
        if not labels_at_ts:
            continue

        node_ids = list(labels_at_ts.keys())
        true_labels = np.array([labels_at_ts[n] for n in node_ids], dtype=np.float32)

        feats = state.get_combined_features(node_ids, t_val)
        feats[:, num_classes:] = (feats[:, num_classes:] - struct_mean) / struct_std

        with torch.no_grad():
            logits = model(torch.FloatTensor(feats).to(device))
            pred_probs = F.softmax(logits, dim=-1).cpu().numpy()

        result_dict = evaluator.eval({
            "y_true": true_labels,
            "y_pred": pred_probs,
            "eval_metric": [eval_metric],
        })
        metric_list.append(result_dict[eval_metric])

        # Update labels for next timestep prediction
        state.update_labels(node_ids, true_labels)

    final_metric = np.mean(metric_list) if metric_list else 0.0
    print(f"\n  Test {eval_metric.upper()}: {final_metric:.6f}")
    print(f"  Params: {n_params:,}")
    print(f"{'='*60}")

    result = {
        "dataset": args.dataset,
        "model": "DTLP-Node-v2",
        eval_metric: float(final_metric),
        "seed": args.seed,
        "n_params": n_params,
        "ema_alpha": args.ema_alpha,
    }
    os.makedirs("/data/chenlibin/TGB2/results", exist_ok=True)
    out_path = get_safe_result_path(
        "/data/chenlibin/TGB2/results",
        f"{args.dataset}_nodev2_seed{args.seed}",
    )
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--ema_alpha", type=float, default=0.3,
                        help="EMA decay for label history")
    args = parser.parse_args()
    train_and_evaluate(args)
