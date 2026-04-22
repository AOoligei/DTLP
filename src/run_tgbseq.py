"""
DTLP on TGB-Seq datasets (ML-20M, Taobao, Yelp, GoogleLocal, Flickr, YouTube, Patent, WikiLink).

Uses TGB-Seq evaluation protocol: provided historical negatives, MRR metric.
"""

import sys
sys.path.insert(0, "/data/chenlibin/TGB2/src")
sys.path.insert(0, "/data/chenlibin/CRAFT_v2")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
import os
import pickle
import argparse
from collections import defaultdict


def compute_tie_aware_mrr(scores):
    """Match the official TGB-Seq evaluator's tie handling."""
    scores = np.asarray(scores, dtype=np.float32)
    pos_score = scores[0]
    neg_scores = scores[1:]
    optimistic_rank = float((neg_scores > pos_score).sum())
    pessimistic_rank = float((neg_scores >= pos_score).sum())
    return 1.0 / (0.5 * (optimistic_rank + pessimistic_rank) + 1.0)


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
    if args.n_neg < 1:
        raise ValueError("--n_neg must be >= 1.")
    if args.max_train < 1:
        raise ValueError("--max_train must be >= 1.")


class HybridTKGScorer(nn.Module):
    def __init__(self, num_relations, entity_feat_dim, pair_feat_dim,
                 rel_dim=32, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.rel_emb = nn.Embedding(max(num_relations, 1), rel_dim)
        self.entity_scorer = nn.Sequential(
            nn.Linear(entity_feat_dim + rel_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        self.src_scorer = nn.Sequential(
            nn.Linear(entity_feat_dim + rel_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        self.pair_scorer = nn.Sequential(
            nn.Linear(pair_feat_dim + rel_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, src_feat, dst_feat, rels, pair_feat):
        rel_emb = self.rel_emb(rels)
        dst_input = torch.cat([dst_feat, rel_emb], dim=-1)
        dst_score = self.entity_scorer(dst_input).squeeze(-1)
        src_input = torch.cat([src_feat, rel_emb], dim=-1)
        src_score = self.src_scorer(src_input).squeeze(-1)
        pair_input = torch.cat([pair_feat, rel_emb], dim=-1)
        pair_score = self.pair_scorer(pair_input).squeeze(-1)
        return dst_score + src_score + pair_score


class IncrementalState:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.out_degree = np.zeros(num_nodes, dtype=np.int32)
        self.in_degree = np.zeros(num_nodes, dtype=np.int32)
        self.last_time_src = np.full(num_nodes, -1e9)
        self.last_time_dst = np.full(num_nodes, -1e9)
        self.n_relations = np.zeros(num_nodes, dtype=np.int32)
        self.n_neighbors = np.zeros(num_nodes, dtype=np.int32)
        self._rel_sets = [set() for _ in range(num_nodes)]
        self._nbr_sets = [set() for _ in range(num_nodes)]
        self.triple_count = defaultdict(int)
        self.pair_count = defaultdict(int)
        self.rel_dst_count = defaultdict(int)
        self.src_dsts = defaultdict(set)
        self.src_rel_dsts = defaultdict(set)
        self.current_time = 0.0
        self.num_edges = 0

    def update(self, src, dst, t, rel):
        self.out_degree[src] += 1
        self.in_degree[dst] += 1
        self.last_time_src[src] = t
        self.last_time_dst[dst] = t
        if rel >= 0:
            if rel not in self._rel_sets[src]:
                self._rel_sets[src].add(rel)
                self.n_relations[src] += 1
            if rel not in self._rel_sets[dst]:
                self._rel_sets[dst].add(rel)
                self.n_relations[dst] += 1
        if dst not in self._nbr_sets[src]:
            self._nbr_sets[src].add(dst)
            self.n_neighbors[src] += 1
        if src not in self._nbr_sets[dst]:
            self._nbr_sets[dst].add(src)
            self.n_neighbors[dst] += 1
        self.triple_count[(src, dst, rel)] += 1
        self.pair_count[(src, dst)] += 1
        self.rel_dst_count[(rel, dst)] += 1
        self.src_dsts[src].add(dst)
        self.src_rel_dsts[(src, rel)].add(dst)
        self.current_time = t
        self.num_edges += 1

    def get_entity_features(self, node, t):
        src_rec = t - self.last_time_src[node]
        dst_rec = t - self.last_time_dst[node]
        return np.array([
            np.log1p(self.out_degree[node]),
            np.log1p(self.in_degree[node]),
            np.log1p(max(0, src_rec)) if src_rec < 1e8 else -1.0,
            np.log1p(max(0, dst_rec)) if dst_rec < 1e8 else -1.0,
            np.log1p(self.n_relations[node]),
            np.log1p(self.n_neighbors[node]),
        ], dtype=np.float32)

    def get_all_entity_features(self, t):
        feats = np.zeros((self.num_nodes, 6), dtype=np.float32)
        feats[:, 0] = np.log1p(self.out_degree)
        feats[:, 1] = np.log1p(self.in_degree)
        src_rec = t - self.last_time_src
        dst_rec = t - self.last_time_dst
        valid_src = src_rec < 1e8
        valid_dst = dst_rec < 1e8
        feats[:, 2] = np.where(valid_src, np.log1p(np.maximum(0, src_rec)), -1.0)
        feats[:, 3] = np.where(valid_dst, np.log1p(np.maximum(0, dst_rec)), -1.0)
        feats[:, 4] = np.log1p(self.n_relations)
        feats[:, 5] = np.log1p(self.n_neighbors)
        return feats

    def get_nodes_entity_features(self, nodes, t):
        nodes = np.asarray(nodes, dtype=np.int64)
        feats = np.zeros((len(nodes), 6), dtype=np.float32)
        feats[:, 0] = np.log1p(self.out_degree[nodes])
        feats[:, 1] = np.log1p(self.in_degree[nodes])
        src_rec = t - self.last_time_src[nodes]
        dst_rec = t - self.last_time_dst[nodes]
        valid_src = src_rec < 1e8
        valid_dst = dst_rec < 1e8
        feats[:, 2] = np.where(valid_src, np.log1p(np.maximum(0, src_rec)), -1.0)
        feats[:, 3] = np.where(valid_dst, np.log1p(np.maximum(0, dst_rec)), -1.0)
        feats[:, 4] = np.log1p(self.n_relations[nodes])
        feats[:, 5] = np.log1p(self.n_neighbors[nodes])
        return feats

    def get_pair_features(self, src, dst, rel):
        return np.array([
            np.log1p(self.triple_count.get((src, dst, rel), 0)),
            np.log1p(self.pair_count.get((src, dst), 0)),
            np.log1p(self.pair_count.get((dst, src), 0)),
            np.log1p(self.rel_dst_count.get((rel, dst), 0)),
        ], dtype=np.float32)


def train_and_evaluate(args):
    validate_args(args)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # 1. Load via TGB-Seq API
    print(f"Loading {args.dataset} from TGB-Seq...")
    from tgb_seq.LinkPred.dataloader import TGBSeqLoader
    data_root = "/data/chenlibin/tgb_seq_datasets"
    loader = TGBSeqLoader(name=args.dataset, root=data_root)
    df = loader._edgelist_df

    sources = df['src'].values.astype(np.int64)
    destinations = df['dst'].values.astype(np.int64)
    timestamps = df['time'].values.astype(np.float64)
    edge_types = np.zeros(len(sources), dtype=np.int64)

    num_nodes = int(max(sources.max(), destinations.max())) + 1
    num_relations = 1
    total = len(sources)

    # Use TGB-Seq official split if available, else 70/15/15
    split = loader.split
    if split is not None:
        split = split.astype(np.int64, copy=False)
        train_mask = split == 0
        train_indices_full = np.flatnonzero(train_mask)
        if train_indices_full.size == 0:
            raise ValueError(f"{args.dataset} has no split=0 training edges.")
        train_end = int(train_indices_full[-1]) + 1
        if not np.all(train_mask[:train_end]):
            raise ValueError(
                f"{args.dataset} has non-contiguous training rows; run_tgbseq.py expects all split=0 edges to be a prefix."
            )
        n_val_rows = int((split == 1).sum())
        n_test_rows = int((split == 2).sum())
        n_context_rows = int((split < 0).sum())
        val_end = train_end + n_val_rows
    else:
        train_end = int(total * 0.70)
        val_end = int(total * 0.85)
        n_val_rows = val_end - train_end
        n_test_rows = total - val_end
        n_context_rows = 0

    print(f"  Nodes: {num_nodes}, Edges: {total}")
    print(f"  Train: {train_end}, Val: {n_val_rows}, Test: {n_test_rows}")
    if n_context_rows > 0:
        print(f"  Context-only post-train edges: {n_context_rows}")

    # 2. Build state (with cache)
    state = IncrementalState(num_nodes)
    warmup_end = train_end * 4 // 5

    cache_dir = "/data/chenlibin/TGB2/datasets"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"tgbseq_{args.dataset}_warmup{warmup_end}_state.pkl")

    if os.path.exists(cache_path):
        print(f"\nLoading cached state...")
        t0 = time.time()
        with open(cache_path, "rb") as f:
            state = pickle.load(f)
        print(f"  Loaded in {time.time()-t0:.1f}s")
    else:
        print(f"\nWarming up on first {warmup_end} edges...")
        t0 = time.time()
        for i in range(warmup_end):
            state.update(int(sources[i]), int(destinations[i]),
                         float(timestamps[i]), 0)
            if (i + 1) % 1000000 == 0:
                print(f"    {i+1}/{warmup_end} ({time.time()-t0:.1f}s)")
        print(f"  Done in {time.time()-t0:.1f}s")
        with open(cache_path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Saved cache.")

    # Per-node recurrence rate (for adaptive_alpha)
    node_rec_rate = {}
    if args.adaptive_alpha and args.hard_neg_ratio > 0:
        print("\nComputing per-node recurrence rates for adaptive_alpha...")
        for s, dsts in state.src_dsts.items():
            if len(dsts) == 0:
                node_rec_rate[s] = 0.0
            else:
                n_rec = sum(1 for d in dsts if state.pair_count.get((s, d), 0) > 1)
                node_rec_rate[s] = n_rec / len(dsts)
        rates = list(node_rec_rate.values())
        print(f"  node_rec_rate: mean={np.mean(rates):.4f} median={np.median(rates):.4f} min={np.min(rates):.4f} max={np.max(rates):.4f}")
        print(f"  Alpha range: [{args.hard_neg_ratio * (1 - np.max(rates)):.4f}, {args.hard_neg_ratio * (1 - np.min(rates)):.4f}]")

    # 3. Collect training data (subsample for very large datasets)
    n_train_full = train_end - warmup_end
    max_train = min(n_train_full, args.max_train)
    if max_train < n_train_full:
        print(f"\nSubsampling {max_train}/{n_train_full} training edges")
        train_indices = np.sort(np.random.choice(n_train_full, max_train, replace=False))
    else:
        train_indices = np.arange(n_train_full)

    n_train = len(train_indices)
    if n_train < 1:
        raise ValueError(
            f"{args.dataset} has no trainable edges after warmup/subsampling (train_end={train_end}, warmup_end={warmup_end}, max_train={args.max_train})."
        )
    n_neg = args.n_neg
    print(f"\nCollecting {n_train} trainable edges...")
    t0 = time.time()

    entity_feat_dim = 6
    pair_feat_dim = 4

    train_src_feats = np.zeros((n_train, entity_feat_dim), dtype=np.float32)
    train_dst_feats = np.zeros((n_train, entity_feat_dim), dtype=np.float32)
    train_pair_feats = np.zeros((n_train, pair_feat_dim), dtype=np.float32)
    train_neg_dst_feats = np.zeros((n_train, n_neg, entity_feat_dim), dtype=np.float32)
    train_neg_pair_feats = np.zeros((n_train, n_neg, pair_feat_dim), dtype=np.float32)
    train_rels = np.zeros(n_train, dtype=np.int64)

    # We must process ALL edges sequentially for state updates
    feat_idx = 0
    train_set = set(train_indices.tolist())
    for i in range(n_train_full):
        real_idx = warmup_end + i
        s = int(sources[real_idx])
        d = int(destinations[real_idx])
        t_i = float(timestamps[real_idx])

        if i in train_set:
            train_src_feats[feat_idx] = state.get_entity_features(s, t_i)
            train_dst_feats[feat_idx] = state.get_entity_features(d, t_i)
            train_pair_feats[feat_idx] = state.get_pair_features(s, d, 0)
            train_rels[feat_idx] = 0

            # Hard-neg + adaptive_alpha
            if args.hard_neg_ratio > 0:
                hist_nbrs = list(state.src_dsts.get(s, set()) - {d})
            else:
                hist_nbrs = []
            if args.adaptive_alpha and args.hard_neg_ratio > 0:
                alpha_s = args.hard_neg_ratio * (1 - node_rec_rate.get(s, 0.0))
            else:
                alpha_s = args.hard_neg_ratio

            for k in range(n_neg):
                use_hard = (alpha_s > 0 and len(hist_nbrs) > 0 and np.random.random() < alpha_s)
                if use_hard:
                    nd = hist_nbrs[np.random.randint(len(hist_nbrs))]
                else:
                    nd = np.random.randint(num_nodes)
                    while nd == d:
                        nd = np.random.randint(num_nodes)
                train_neg_dst_feats[feat_idx, k] = state.get_entity_features(nd, t_i)
                train_neg_pair_feats[feat_idx, k] = state.get_pair_features(s, nd, 0)
            feat_idx += 1

        state.update(s, d, t_i, 0)

        if (i + 1) % 200000 == 0:
            print(f"    {i+1}/{n_train_full} ({time.time()-t0:.1f}s)")

    print(f"  Done in {time.time()-t0:.1f}s")

    # Normalization
    all_ent = np.concatenate([train_src_feats, train_dst_feats,
                               train_neg_dst_feats.reshape(-1, entity_feat_dim)])
    ent_mean = all_ent.mean(0)
    ent_std = all_ent.std(0) + 1e-8
    all_pf = np.concatenate([train_pair_feats,
                              train_neg_pair_feats.reshape(-1, pair_feat_dim)])
    pair_mean = all_pf.mean(0)
    pair_std = all_pf.std(0) + 1e-8

    train_src_feats = (train_src_feats - ent_mean) / ent_std
    train_dst_feats = (train_dst_feats - ent_mean) / ent_std
    train_pair_feats = (train_pair_feats - pair_mean) / pair_std
    train_neg_dst_feats = (train_neg_dst_feats - ent_mean) / ent_std
    train_neg_pair_feats = (train_neg_pair_feats - pair_mean) / pair_std

    # 4. Model
    model = HybridTKGScorer(
        num_relations, entity_feat_dim, pair_feat_dim,
        rel_dim=args.rel_dim, hidden_dim=args.hidden_dim, dropout=args.dropout
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    t_src_f = torch.FloatTensor(train_src_feats).to(device)
    t_dst_f = torch.FloatTensor(train_dst_feats).to(device)
    t_pair_f = torch.FloatTensor(train_pair_feats).to(device)
    t_neg_dst_f = torch.FloatTensor(train_neg_dst_feats).to(device)
    t_neg_pair_f = torch.FloatTensor(train_neg_pair_feats).to(device)
    t_rels = torch.LongTensor(train_rels).to(device)

    # Ablation
    if args.ablation == 'entity_only':
        t_pair_f.zero_()
        t_neg_pair_f.zero_()
        print("  [ABLATION] entity_only: pair features zeroed")
    elif args.ablation == 'pair_only':
        t_src_f.zero_()
        t_dst_f.zero_()
        t_neg_dst_f.zero_()
        print("  [ABLATION] pair_only: entity features zeroed")

    # 5. Train
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
            pos_scores = model(t_src_f[idx], t_dst_f[idx], t_rels[idx], t_pair_f[idx])
            neg_scores_list = []
            for k in range(n_neg):
                ns = model(t_src_f[idx], t_neg_dst_f[idx, k], t_rels[idx], t_neg_pair_f[idx, k])
                neg_scores_list.append(ns)
            neg_scores = torch.stack(neg_scores_list, dim=1)
            if args.loss == 'bpr':
                diff = pos_scores.unsqueeze(1) - neg_scores
                loss = -F.logsigmoid(diff).mean()
            elif args.loss == 'ce':
                all_scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
                targets = torch.zeros(len(idx), dtype=torch.long, device=device)
                loss = F.cross_entropy(all_scores, targets)
            elif args.loss == 'margin':
                margin = 0.1
                diff = pos_scores.unsqueeze(1) - neg_scores
                loss = torch.clamp(margin - diff, min=0).mean()
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
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f} (best: ep{best_epoch} {best_loss:.4f})")
        if patience_counter >= 15 and epoch >= 10:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    if best_epoch in checkpoints:
        model.load_state_dict({k: v.to(device) for k, v in checkpoints[best_epoch].items()})
        print(f"  Restored best model from epoch {best_epoch} (loss={best_loss:.4f})")

    # 6. Evaluate using TGB-Seq test negatives
    print(f"\n{'='*60}")
    print("Evaluating (TGB-Seq protocol: historical negatives, MRR)...")
    model.eval()

    # Load test negatives
    test_ns = loader.negative_samples  # shape: (num_test_queries, num_neg)
    if test_ns is None:
        raise FileNotFoundError(
            f"Missing TGB-Seq test negatives for {args.dataset}; expected a .npy file under {data_root}/{args.dataset}."
        )

    # For datasets with post-train context/eval blocks (Patent repeats -1 -> 1 -> 2
    # after the initial split=0 prefix), iterate ALL post-train edges chronologically:
    # - split=-1 or split=1: update state only (no scoring)
    # - split=2: score (test query), then update state
    if split is not None:
        test_indices = np.flatnonzero(split == 2)
        post_train_indices = np.flatnonzero(split != 0)
    else:
        test_indices = np.arange(val_end, total)
        post_train_indices = np.arange(train_end, total)

    num_test = len(test_indices)
    print(f"  Test queries: {num_test}, Test NS: {test_ns.shape[0]}")
    if test_ns.shape[0] != num_test:
        raise ValueError(
            f"Test negative samples ({test_ns.shape[0]}) do not match split=2 queries ({num_test}) for {args.dataset}."
        )

    # Build mapping: global_index → test_ns_index (for split=2 edges)
    test_ns_map = {int(test_indices[j]): j for j in range(num_test)}
    if len(post_train_indices) > 1:
        post_train_order = np.lexsort((post_train_indices, timestamps[post_train_indices]))
        post_train_indices = post_train_indices[post_train_order]

    mrr_list = []
    t0 = time.time()
    n_scored = 0

    # Iterate ALL post-train edges chronologically
    if len(post_train_indices) > 0:
        for gi in post_train_indices:
            src_i = int(sources[gi])
            dst_i = int(destinations[gi])
            t_i = float(timestamps[gi])

            # If not a test query, just update state
            if gi not in test_ns_map:
                state.update(src_i, dst_i, t_i, 0)
                continue

            # This is a test query (split=2): score first, then update
            ns_idx = test_ns_map[gi]

            neg_dsts = test_ns[ns_idx]
            neg_dsts = neg_dsts[neg_dsts >= 0]  # filter padding

            all_dsts = np.concatenate([[dst_i], neg_dsts.astype(int)])
            n_cand = len(all_dsts)

            src_feat = state.get_entity_features(src_i, t_i)
            src_feat_n = (src_feat - ent_mean) / ent_std
            dst_feats_raw = state.get_nodes_entity_features(all_dsts, t_i)
            dst_feats = (dst_feats_raw - ent_mean) / ent_std
            pair_feats = np.zeros((n_cand, pair_feat_dim), dtype=np.float32)
            for j, cand_dst in enumerate(all_dsts):
                pair_feats[j] = state.get_pair_features(src_i, int(cand_dst), 0)
            pair_feats = (pair_feats - pair_mean) / pair_std

            if args.ablation == 'entity_only':
                pair_feats[:] = 0
            elif args.ablation == 'pair_only':
                src_feat_n[:] = 0
                dst_feats[:] = 0

            with torch.no_grad():
                src_t = torch.FloatTensor(np.tile(src_feat_n, (n_cand, 1))).to(device)
                dst_t = torch.FloatTensor(dst_feats).to(device)
                rel_t = torch.zeros(n_cand, dtype=torch.long, device=device)
                pair_t = torch.FloatTensor(pair_feats).to(device)
                scores = model(src_t, dst_t, rel_t, pair_t).cpu().numpy()

            mrr_list.append(compute_tie_aware_mrr(scores))
            n_scored += 1

            state.update(src_i, dst_i, t_i, 0)

            if n_scored % 50000 == 0:
                print(f"    {n_scored}/{num_test}, MRR: {np.mean(mrr_list):.4f} ({time.time()-t0:.1f}s)")

    final_mrr = float(np.mean(mrr_list)) if mrr_list else 0.0
    print(f"\n  Test MRR: {final_mrr:.6f}")
    print(f"  Params: {n_params:,}")

    result = {
        "dataset": args.dataset,
        "model": "DTLP",
        "protocol": "tgbseq_historical_neg",
        "mrr": float(final_mrr),
        "seed": args.seed,
        "n_params": n_params,
        "n_test": num_test,
        "ablation": args.ablation,
        "loss": args.loss,
        "hard_neg_ratio": args.hard_neg_ratio,
        "adaptive_alpha": args.adaptive_alpha,
    }
    os.makedirs("/data/chenlibin/TGB2/results", exist_ok=True)
    stem = f"tgbseq_{args.dataset}"
    if args.ablation != "none":
        stem += f"_{args.ablation}"
    if args.loss != "bpr":
        stem += f"_{args.loss}"
    if args.hard_neg_ratio > 0:
        stem += f"_hn{args.hard_neg_ratio}"
        if args.adaptive_alpha:
            stem += "_adapt"
    stem += f"_seed{args.seed}"
    out_path = get_safe_result_path(
        "/data/chenlibin/TGB2/results",
        stem,
    )
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--n_neg", type=int, default=5)
    parser.add_argument("--max_train", type=int, default=2000000,
                        help="Max training edges (subsample if larger)")
    parser.add_argument("--rel_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--ablation", type=str, default="none",
                        choices=["none", "entity_only", "pair_only"])
    parser.add_argument("--loss", type=str, default="bpr", choices=["bpr", "ce", "margin"])
    parser.add_argument("--hard_neg_ratio", type=float, default=0.0,
                        help="Fraction of negatives sampled from historical neighbors (0=off)")
    parser.add_argument("--adaptive_alpha", action="store_true",
                        help="Per-source adaptive alpha: alpha(s) = hard_neg_ratio * (1 - node_rec_rate(s))")
    args = parser.parse_args()
    train_and_evaluate(args)
