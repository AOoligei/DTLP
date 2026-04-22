"""
TKG Link Prediction: Hybrid Decomposable + Sparse Pair Scoring
WITH ADAPTIVE PAIR GATING

Two gating modes:
  --gate global:  single learnable α, pair_score *= sigmoid(α)
  --gate source:  gate = sigmoid(MLP(src_features)), repurposes dead src_scorer

Both use L1 penalty (--gate_l1) to push gate toward 0 by default,
so pair features are only used when the training signal justifies it.

Hypothesis: on low-recurrence datasets (tgbl-review), gate → 0;
on high-recurrence datasets (tgbl-wiki), gate → 1.
"""

import sys
sys.path.insert(0, "/data/chenlibin/TGB2/src")
sys.path.insert(0, "/data/chenlibin/CRAFT_v2")  # Use vendored tgb (pip tgb has csv_to_pd_data_sc bug on coin)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tgb.linkproppred.dataset import LinkPropPredDataset
from tgb.linkproppred.evaluate import Evaluator
import time
import json
import os
import pickle
import hashlib
import argparse
from collections import defaultdict


class GatedHybridTKGScorer(nn.Module):
    """Hybrid scorer with adaptive pair feature gating.

    Score(src, rel, dst) = dst_score(dst, rel) + gate * pair_score(src, rel, dst)

    Gate modes:
      'none':   gate = 1 (original behavior)
      'global': gate = sigmoid(alpha), alpha is a single learnable parameter
      'source': gate = sigmoid(MLP(src_features)), per-query adaptive gate
    """

    def __init__(self, num_relations, entity_feat_dim, pair_feat_dim=4,
                 rel_dim=32, hidden_dim=64, dropout=0.1, gate_type='none'):
        super().__init__()
        self.gate_type = gate_type

        # Relation embeddings
        self.rel_emb = nn.Embedding(num_relations, rel_dim)
        nn.init.xavier_uniform_(self.rel_emb.weight)

        # Entity feature scorer: entity_feats + rel_emb → scalar
        self.entity_scorer = nn.Sequential(
            nn.Linear(entity_feat_dim + rel_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Pair feature scorer: pair_feats + rel_emb → scalar
        self.pair_scorer = nn.Sequential(
            nn.Linear(pair_feat_dim + rel_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Gate components
        if gate_type == 'global':
            # Option 2: single learnable scalar, initialized to 0 → sigmoid(0) = 0.5
            self.pair_alpha = nn.Parameter(torch.tensor(0.0))
        elif gate_type == 'source':
            # Option 1: repurpose src_scorer as gate MLP
            # Takes src entity features → scalar gate in [0, 1]
            self.gate_mlp = nn.Sequential(
                nn.Linear(entity_feat_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )

        # Keep src_scorer for non-gated mode (dead under BPR but for compatibility)
        if gate_type == 'none':
            self.src_scorer = nn.Sequential(
                nn.Linear(entity_feat_dim + rel_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

    def compute_gate(self, src_feats=None):
        """Compute pair feature gate value.

        Returns:
            gate: scalar or (B,) tensor in [0, 1]
        """
        if self.gate_type == 'none':
            return torch.tensor(1.0)
        elif self.gate_type == 'global':
            return torch.sigmoid(self.pair_alpha)
        elif self.gate_type == 'source':
            return torch.sigmoid(self.gate_mlp(src_feats).squeeze(-1))

    def forward(self, src_feats, dst_feats, rel_ids, pair_feats):
        """Score specific (src, rel, dst) triplets.

        Returns:
            scores: (B,)
            gate: scalar or (B,) gate values (for L1 penalty)
        """
        rel = self.rel_emb(rel_ids)  # (B, rel_dim)

        # Destination score (entity features)
        dst_input = torch.cat([dst_feats, rel], dim=-1)
        dst_score = self.entity_scorer(dst_input).squeeze(-1)  # (B,)

        # Pair score
        pair_input = torch.cat([pair_feats, rel], dim=-1)
        pair_score = self.pair_scorer(pair_input).squeeze(-1)  # (B,)

        # Gate
        gate = self.compute_gate(src_feats)

        if self.gate_type == 'none':
            # Original: dst + src + pair
            src_input = torch.cat([src_feats, rel], dim=-1)
            src_score = self.src_scorer(src_input).squeeze(-1)
            return dst_score + src_score + pair_score, gate
        elif self.gate_type == 'global':
            # gate is a scalar
            return dst_score + gate * pair_score, gate
        elif self.gate_type == 'source':
            # gate is (B,) per-query
            return dst_score + gate * pair_score, gate


class IncrementalState:
    """Track graph state incrementally for feature computation."""

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
        feats[:, 2] = np.where(src_rec < 1e8, np.log1p(np.maximum(0, src_rec)), -1.0)
        feats[:, 3] = np.where(dst_rec < 1e8, np.log1p(np.maximum(0, dst_rec)), -1.0)
        feats[:, 4] = np.log1p(self.n_relations)
        feats[:, 5] = np.log1p(self.n_neighbors)
        return feats

    def get_pair_features(self, src, dst, rel):
        return np.array([
            np.log1p(self.triple_count.get((src, dst, rel), 0)),
            np.log1p(self.pair_count.get((src, dst), 0)),
            np.log1p(self.pair_count.get((dst, src), 0)),
            np.log1p(self.rel_dst_count.get((rel, dst), 0)),
        ], dtype=np.float32)


def train_and_evaluate(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # 1. Load
    print(f"Loading {args.dataset}...")
    dataset = LinkPropPredDataset(name=args.dataset, root="datasets")
    data = dataset.full_data
    num_nodes = int(max(data['sources'].max(), data['destinations'].max())) + 1
    edge_types = data.get('edge_type', np.full(len(data['sources']), 0))
    num_relations = int(edge_types.max()) + 1

    sources = data['sources']
    destinations = data['destinations']
    timestamps = data['timestamps']
    ets = edge_types.astype(np.int64)

    train_end = int(dataset.train_mask.sum())
    val_end = train_end + int(dataset.val_mask.sum())
    total = len(sources)

    print(f"  Nodes: {num_nodes}, Relations: {num_relations}, Edges: {total}")
    print(f"  Train: {train_end}, Val: {val_end-train_end}, Test: {total-val_end}")

    # 2. Build state (with caching)
    cache_dir = os.path.join("datasets", f"{args.dataset}_cache")
    os.makedirs(cache_dir, exist_ok=True)

    warmup_end = int(train_end * args.warmup_ratio)
    state_cache = os.path.join(cache_dir, f"state_w{args.warmup_ratio}.pkl")

    if os.path.exists(state_cache):
        print(f"\nLoading cached warmup state from {state_cache}...")
        t0 = time.time()
        with open(state_cache, 'rb') as f:
            state = pickle.load(f)
        print(f"  Done in {time.time()-t0:.1f}s (skipped {warmup_end} edge warmup)")
    else:
        state = IncrementalState(num_nodes)
        print(f"\nWarming up on first {warmup_end} edges...")
        t0 = time.time()
        for i in range(warmup_end):
            state.update(int(sources[i]), int(destinations[i]),
                         float(timestamps[i]), int(ets[i]))
        print(f"  Done in {time.time()-t0:.1f}s")
        with open(state_cache, 'wb') as f:
            pickle.dump(state, f)
        print(f"  Saved warmup state to {state_cache}")

    # Compute pair recurrence rate from warmup data
    n_unique_pairs = len(state.pair_count)
    n_recurrent_pairs = sum(1 for v in state.pair_count.values() if v > 1)
    pair_rec_rate = n_recurrent_pairs / max(n_unique_pairs, 1)
    print(f"  Pair recurrence rate: {pair_rec_rate:.4f} ({n_recurrent_pairs}/{n_unique_pairs})")

    # Per-node recurrence rates (for adaptive alpha)
    node_rec_rate = {}
    if args.adaptive_alpha and args.hard_neg_ratio > 0:
        for s, dsts in state.src_dsts.items():
            if len(dsts) == 0:
                node_rec_rate[s] = 0.0
            else:
                n_rec = sum(1 for d in dsts if state.pair_count.get((s, d), 0) > 1)
                node_rec_rate[s] = n_rec / len(dsts)
        rates = list(node_rec_rate.values())
        print(f"  Adaptive alpha: node_rec_rate distribution: "
              f"mean={np.mean(rates):.4f}, median={np.median(rates):.4f}, "
              f"min={np.min(rates):.4f}, max={np.max(rates):.4f}")
        print(f"  Alpha range: [{args.hard_neg_ratio * (1 - np.max(rates)):.4f}, "
              f"{args.hard_neg_ratio * (1 - np.min(rates)):.4f}]")

    # Auto gate: if recurrence rate is low, disable pair features entirely
    auto_gate_value = None
    if args.gate == 'auto':
        AUTO_THRESHOLD = 0.05  # 5% recurrence threshold
        if pair_rec_rate < AUTO_THRESHOLD:
            auto_gate_value = 0.0  # entity-only mode
            print(f"  AUTO GATE: pair_rec_rate={pair_rec_rate:.4f} < {AUTO_THRESHOLD} → gate=0 (entity-only)")
        else:
            auto_gate_value = 1.0  # full model
            print(f"  AUTO GATE: pair_rec_rate={pair_rec_rate:.4f} >= {AUTO_THRESHOLD} → gate=1 (full)")
        # Use 'none' gate type internally (no learnable params)
        args.gate = 'none'
        # We'll override gate in forward pass

    # Type-constrained negative candidates
    type_to_dsts = defaultdict(set)
    for i in range(train_end):
        type_to_dsts[int(ets[i])].add(int(destinations[i]))
    type_to_dsts_arr = {et: np.array(list(d)) for et, d in type_to_dsts.items()}

    # 3. Collect training data (with caching)
    n_train = train_end - warmup_end
    n_neg = args.n_neg
    entity_feat_dim = 6
    pair_feat_dim = 4

    hn_tag = f"hn{args.hard_neg_ratio}_adapt" if args.adaptive_alpha else f"hn{args.hard_neg_ratio}"
    feat_cache = os.path.join(cache_dir, f"feats_s{args.seed}_{hn_tag}_neg{n_neg}.npz")

    if os.path.exists(feat_cache):
        print(f"\nLoading cached features from {feat_cache}...")
        t0 = time.time()
        cached = np.load(feat_cache)
        train_src_feats = cached['src_feats']
        train_dst_feats = cached['dst_feats']
        train_pair_feats = cached['pair_feats']
        train_neg_dst_feats = cached['neg_dst_feats']
        train_neg_pair_feats = cached['neg_pair_feats']
        train_rels = cached['rels']
        # Replay state updates for training edges (needed for eval)
        for i in range(n_train):
            idx = warmup_end + i
            state.update(int(sources[idx]), int(destinations[idx]),
                         float(timestamps[idx]), int(ets[idx]))
        print(f"  Done in {time.time()-t0:.1f}s (skipped feature collection)")
    else:
        print(f"\nCollecting {n_train} trainable edges...")
        t0 = time.time()

        train_src_feats = np.zeros((n_train, entity_feat_dim), dtype=np.float32)
        train_dst_feats = np.zeros((n_train, entity_feat_dim), dtype=np.float32)
        train_pair_feats = np.zeros((n_train, pair_feat_dim), dtype=np.float32)
        train_neg_dst_feats = np.zeros((n_train, n_neg, entity_feat_dim), dtype=np.float32)
        train_neg_pair_feats = np.zeros((n_train, n_neg, pair_feat_dim), dtype=np.float32)
        train_rels = np.zeros(n_train, dtype=np.int64)

        # Set seed for reproducible negative sampling
        np.random.seed(args.seed)
        for i in range(n_train):
            idx = warmup_end + i
            s = int(sources[idx])
            d = int(destinations[idx])
            t_i = float(timestamps[idx])
            r = int(ets[idx])

            train_src_feats[i] = state.get_entity_features(s, t_i)
            train_dst_feats[i] = state.get_entity_features(d, t_i)
            train_pair_feats[i] = state.get_pair_features(s, d, r)
            train_rels[i] = r

            candidates = type_to_dsts_arr.get(r, None)
            # Hard-negative sampling: sample from historical neighbors
            hist_nbrs = list(state.src_dsts.get(s, set()) - {d})
            # Adaptive alpha: alpha(s) = alpha_max * (1 - node_rec_rate(s))
            if args.adaptive_alpha and args.hard_neg_ratio > 0:
                alpha_s = args.hard_neg_ratio * (1 - node_rec_rate.get(s, 0.0))
            else:
                alpha_s = args.hard_neg_ratio
            for k in range(n_neg):
                use_hard = (alpha_s > 0 and
                           len(hist_nbrs) > 0 and
                           np.random.random() < alpha_s)
                if use_hard:
                    nd = hist_nbrs[np.random.randint(len(hist_nbrs))]
                elif candidates is not None and len(candidates) > 1:
                    nd = candidates[np.random.randint(len(candidates))]
                    while nd == d:
                        nd = candidates[np.random.randint(len(candidates))]
                else:
                    nd = np.random.randint(num_nodes)
                nd = int(nd)
                train_neg_dst_feats[i, k] = state.get_entity_features(nd, t_i)
                train_neg_pair_feats[i, k] = state.get_pair_features(s, nd, r)

            state.update(s, d, t_i, r)

            if (i + 1) % 50000 == 0:
                print(f"    {i+1}/{n_train} ({time.time()-t0:.1f}s)")

        print(f"  Done in {time.time()-t0:.1f}s")
        np.savez(feat_cache,
                 src_feats=train_src_feats, dst_feats=train_dst_feats,
                 pair_feats=train_pair_feats, neg_dst_feats=train_neg_dst_feats,
                 neg_pair_feats=train_neg_pair_feats, rels=train_rels)
        print(f"  Saved features to {feat_cache}")

    # Normalization
    all_ent_feats = np.concatenate([train_src_feats, train_dst_feats,
                                     train_neg_dst_feats.reshape(-1, entity_feat_dim)])
    ent_mean = all_ent_feats.mean(0)
    ent_std = all_ent_feats.std(0) + 1e-8

    all_pair_feats = np.concatenate([train_pair_feats,
                                      train_neg_pair_feats.reshape(-1, pair_feat_dim)])
    pair_mean = all_pair_feats.mean(0)
    pair_std = all_pair_feats.std(0) + 1e-8

    train_src_feats = (train_src_feats - ent_mean) / ent_std
    train_dst_feats = (train_dst_feats - ent_mean) / ent_std
    train_pair_feats = (train_pair_feats - pair_mean) / pair_std
    train_neg_dst_feats = (train_neg_dst_feats - ent_mean) / ent_std
    train_neg_pair_feats = (train_neg_pair_feats - pair_mean) / pair_std

    # Auto gate: zero out pair features if gate=0
    if auto_gate_value is not None and auto_gate_value == 0.0:
        train_pair_feats[:] = 0.0
        train_neg_pair_feats[:] = 0.0
        print("  AUTO: Zeroed pair features for training (entity-only mode)")

    # 4. Model
    model = GatedHybridTKGScorer(
        num_relations, entity_feat_dim, pair_feat_dim,
        rel_dim=args.rel_dim, hidden_dim=args.hidden_dim, dropout=args.dropout,
        gate_type=args.gate,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    adapt_tag = " (adaptive)" if args.adaptive_alpha else ""
    print(f"\nModel: gate={args.gate}, gate_l1={args.gate_l1}, hard_neg={args.hard_neg_ratio}{adapt_tag}, params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # Tensors
    t_src_f = torch.FloatTensor(train_src_feats).to(device)
    t_dst_f = torch.FloatTensor(train_dst_feats).to(device)
    t_pair_f = torch.FloatTensor(train_pair_feats).to(device)
    t_neg_dst_f = torch.FloatTensor(train_neg_dst_feats).to(device)
    t_neg_pair_f = torch.FloatTensor(train_neg_pair_feats).to(device)
    t_rels = torch.LongTensor(train_rels).to(device)

    # 5. Train
    print(f"\nTraining for {args.epochs} epochs...")
    gate_history = []  # Track gate values per epoch

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        total_loss = 0
        total_gate = 0
        n_batches = 0

        for i in range(0, n_train, args.batch_size):
            idx = perm[i:i + args.batch_size]
            b_src_f = t_src_f[idx]
            b_dst_f = t_dst_f[idx]
            b_pair_f = t_pair_f[idx]
            b_rels = t_rels[idx]

            # Positive scores
            pos_scores, gate = model(b_src_f, b_dst_f, b_rels, b_pair_f)

            # Negative scores
            neg_scores_list = []
            neg_gates = []
            for k in range(n_neg):
                ns, g = model(b_src_f, t_neg_dst_f[idx, k], b_rels,
                              t_neg_pair_f[idx, k])
                neg_scores_list.append(ns)
                neg_gates.append(g)
            neg_scores = torch.stack(neg_scores_list, dim=1)  # (B, n_neg)

            # BPR loss
            diff = pos_scores.unsqueeze(1) - neg_scores
            loss_bpr = -F.logsigmoid(diff).mean()

            # L1 penalty on gate (push toward 0)
            if args.gate != 'none' and args.gate_l1 > 0:
                if args.gate == 'global':
                    gate_val = gate  # scalar
                    loss_l1 = args.gate_l1 * gate_val
                else:
                    # source gate: average over batch
                    all_gates = torch.cat([gate] + neg_gates)
                    gate_val = all_gates.mean()
                    loss_l1 = args.gate_l1 * gate_val
                loss = loss_bpr + loss_l1
            else:
                loss = loss_bpr
                gate_val = gate if isinstance(gate, torch.Tensor) and gate.dim() == 0 else gate.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss_bpr.item()
            if isinstance(gate_val, torch.Tensor):
                total_gate += gate_val.item()
            else:
                total_gate += float(gate_val)
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        avg_gate = total_gate / max(1, n_batches)
        gate_history.append(avg_gate)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}, Gate: {avg_gate:.4f}")

    # Print final gate value
    print(f"\n  Final gate value: {gate_history[-1]:.4f}")
    if args.gate == 'global':
        with torch.no_grad():
            print(f"  Raw alpha: {model.pair_alpha.item():.4f}, sigmoid(alpha): {torch.sigmoid(model.pair_alpha).item():.4f}")

    # 6. Evaluate
    print(f"\n{'='*60}")
    print("Evaluating...")
    model.eval()

    # Process validation edges
    for i in range(train_end, val_end):
        state.update(int(sources[i]), int(destinations[i]),
                     float(timestamps[i]), int(ets[i]))

    dataset.load_test_ns()
    neg_sampler = dataset.negative_sampler
    evaluator = Evaluator(name=args.dataset)
    eval_set = neg_sampler.eval_set.get('test', {})

    test_srcs = sources[val_end:]
    test_dsts = destinations[val_end:]
    test_ts = timestamps[val_end:]
    test_ets = ets[val_end:]
    num_test = len(test_srcs)

    is_linkpred = args.dataset.startswith('tgbl-')
    print(f"  Eval queries: {len(eval_set)}, Test edges: {num_test}, Mode: {'linkpred' if is_linkpred else 'TKG'}")

    # Get final gate value for reporting
    with torch.no_grad():
        if args.gate == 'global':
            final_gate = torch.sigmoid(model.pair_alpha).item()
        elif args.gate == 'source':
            final_gate = 'per-query'
        elif auto_gate_value is not None:
            final_gate = auto_gate_value
        else:
            final_gate = 1.0
    print(f"  Gate value at eval: {final_gate}")

    mrr_list = []
    gate_values_eval = []  # Track per-query gate for source mode
    num_evaluated = 0
    t0 = time.time()

    all_ent_feats_raw = state.get_all_entity_features(float(test_ts[0]))
    all_ent_feats_norm = (all_ent_feats_raw - ent_mean) / ent_std
    recompute_interval = args.recompute_interval
    edges_since_recompute = 0

    for i in range(num_test):
        src_i = int(test_srcs[i])
        dst_i = int(test_dsts[i])
        t_i = float(test_ts[i])
        et_i = int(test_ets[i])

        if is_linkpred:
            key = (src_i, dst_i, int(test_ts[i]))
        else:
            key = (int(test_ts[i]), src_i, et_i)
        if key not in eval_set:
            state.update(src_i, dst_i, t_i, et_i)
            edges_since_recompute += 1
            continue

        if edges_since_recompute >= recompute_interval:
            all_ent_feats_raw = state.get_all_entity_features(t_i)
            all_ent_feats_norm = (all_ent_feats_raw - ent_mean) / ent_std
            edges_since_recompute = 0

        try:
            if is_linkpred:
                neg_batch = neg_sampler.query_batch(
                    np.array([src_i]), np.array([dst_i]),
                    np.array([test_ts[i]]),
                    split_mode='test'
                )
                neg_dsts_j = np.array(neg_batch[0]).astype(int)
            else:
                neg_batch = neg_sampler.query_batch(
                    np.array([src_i]), np.array([dst_i]),
                    np.array([test_ts[i]]),
                    edge_type=np.array([et_i]),
                    split_mode='test'
                )
                neg_dsts_j = neg_batch[0].astype(int)
        except Exception:
            state.update(src_i, dst_i, t_i, et_i)
            edges_since_recompute += 1
            continue

        # Phase 1: Entity-level scores for ALL candidates
        all_dsts = np.concatenate([[dst_i], neg_dsts_j])
        n_cand = len(all_dsts)

        src_feat = state.get_entity_features(src_i, t_i)
        src_feat_n = (src_feat - ent_mean) / ent_std
        dst_feats = all_ent_feats_norm[all_dsts]

        with torch.no_grad():
            dst_t = torch.FloatTensor(dst_feats).to(device)
            rel_t = torch.LongTensor(np.full(n_cand, et_i)).to(device)
            rel_emb_all = model.rel_emb(rel_t)

            dst_input = torch.cat([dst_t, rel_emb_all], dim=-1)
            dst_scores = model.entity_scorer(dst_input).squeeze(-1)
            base_scores = dst_scores.cpu().numpy()

            # Compute gate for this query
            if args.gate == 'global':
                query_gate = torch.sigmoid(model.pair_alpha).item()
            elif args.gate == 'source':
                src_t = torch.FloatTensor(src_feat_n).unsqueeze(0).to(device)
                query_gate = torch.sigmoid(model.gate_mlp(src_t)).item()
                gate_values_eval.append(query_gate)
            else:
                query_gate = 1.0

            # Override with auto gate if set
            if auto_gate_value is not None:
                query_gate = auto_gate_value

        # Phase 2: Sparse pair bonus (gated)
        if query_gate > 0.001:  # Skip pair scoring if gate is essentially closed
            known_dsts_for_src = state.src_dsts.get(src_i, set())
            known_rel_dsts = state.src_rel_dsts.get((src_i, et_i), set())

            cand_with_pair = []
            cand_indices = []
            dst_to_idx = {int(d): j for j, d in enumerate(all_dsts)}
            for d_int, j in dst_to_idx.items():
                has_src_pair = (d_int in known_dsts_for_src) or (d_int in known_rel_dsts)
                has_rel_dst = (et_i, d_int) in state.rel_dst_count
                if has_src_pair or has_rel_dst:
                    cand_with_pair.append(d_int)
                    cand_indices.append(j)

            if cand_with_pair:
                n_pair = len(cand_with_pair)
                pair_feats = np.zeros((n_pair, pair_feat_dim), dtype=np.float32)
                for j, d in enumerate(cand_with_pair):
                    pf = state.get_pair_features(src_i, int(d), et_i)
                    pair_feats[j] = (pf - pair_mean) / pair_std

                with torch.no_grad():
                    rel_t = torch.LongTensor(np.full(n_pair, et_i)).to(device)
                    rel_emb = model.rel_emb(rel_t)
                    pair_input = torch.cat([
                        torch.FloatTensor(pair_feats).to(device), rel_emb
                    ], dim=-1)
                    pair_bonus = model.pair_scorer(pair_input).squeeze(-1).cpu().numpy()

                # Apply gate to pair bonus
                for j, idx in enumerate(cand_indices):
                    base_scores[idx] += query_gate * pair_bonus[j]

        scores_np = base_scores

        input_dict = {
            "y_pred_pos": scores_np[0:1],
            "y_pred_neg": scores_np[1:].reshape(1, -1),
            "eval_metric": ["mrr"],
        }
        result = evaluator.eval(input_dict)
        mrr_list.append(result["mrr"])
        num_evaluated += 1

        state.update(src_i, dst_i, t_i, et_i)
        edges_since_recompute += 1

        if num_evaluated % 2000 == 0 or num_evaluated == 1:
            elapsed = time.time() - t0
            curr_mrr = np.mean(mrr_list)
            gate_info = f", gate={query_gate:.3f}" if (args.gate != 'none' or auto_gate_value is not None) else ""
            print(f"    {num_evaluated}/{len(eval_set)}, MRR: {curr_mrr:.4f} ({elapsed:.1f}s{gate_info})")

    final_mrr = np.mean(mrr_list) if mrr_list else 0
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    gate_label = 'auto' if auto_gate_value is not None else args.gate
    print(f"RESULT: {args.dataset} gated_hybrid MRR = {final_mrr:.6f}")
    print(f"  Gate type: {gate_label}, L1: {args.gate_l1}")
    if auto_gate_value is not None:
        print(f"  Auto gate: {auto_gate_value} (pair_rec_rate={pair_rec_rate:.4f})")
    if args.gate == 'global':
        print(f"  Final gate: sigmoid({model.pair_alpha.item():.4f}) = {torch.sigmoid(model.pair_alpha).item():.4f}")
    elif args.gate == 'source':
        if gate_values_eval:
            avg_gate = np.mean(gate_values_eval)
            std_gate = np.std(gate_values_eval)
            print(f"  Avg source gate: {avg_gate:.4f} ± {std_gate:.4f}")
            print(f"  Gate range: [{min(gate_values_eval):.4f}, {max(gate_values_eval):.4f}]")
    print(f"  {num_evaluated} queries in {elapsed:.1f}s")
    print(f"  Params: {n_params:,}")
    print(f"  Gate history (per epoch): {[f'{g:.3f}' for g in gate_history]}")
    print(f"{'='*60}")

    result_dict = {
        'dataset': args.dataset,
        'model': f'gated_hybrid_{args.gate}',
        'gate_type': args.gate,
        'gate_l1': args.gate_l1,
        'mrr': float(final_mrr),
        'seed': args.seed,
        'n_params': n_params,
        'n_evaluated': num_evaluated,
        'eval_time': elapsed,
        'gate_history': gate_history,
    }
    if args.adaptive_alpha:
        result_dict['adaptive_alpha'] = True
        result_dict['pair_rec_rate'] = pair_rec_rate
    if auto_gate_value is not None:
        result_dict['auto_gate'] = auto_gate_value
        result_dict['pair_rec_rate'] = pair_rec_rate
    elif args.gate == 'global':
        result_dict['final_alpha'] = model.pair_alpha.item()
        result_dict['final_gate'] = torch.sigmoid(model.pair_alpha).item()
    elif args.gate == 'source' and gate_values_eval:
        result_dict['avg_gate'] = float(np.mean(gate_values_eval))
        result_dict['gate_std'] = float(np.std(gate_values_eval))

    os.makedirs("/data/chenlibin/TGB2/results", exist_ok=True)
    hn_suffix = f"_hn{args.hard_neg_ratio}{'_adapt' if args.adaptive_alpha else ''}" if args.hard_neg_ratio > 0 else ""
    out_path = f"/data/chenlibin/TGB2/results/{args.dataset}_gated_{gate_label}_l1{args.gate_l1}{hn_suffix}_seed{args.seed}.json"
    with open(out_path, 'w') as f:
        json.dump(result_dict, f, indent=2)
    print(f"Saved to {out_path}")

    return final_mrr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='tgbl-wiki')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--rel_dim', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--n_neg', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--recompute_interval', type=int, default=500)
    parser.add_argument('--gate', type=str, default='none',
                        choices=['none', 'global', 'source', 'auto'],
                        help='Pair feature gating mode (auto=recurrence-based threshold)')
    parser.add_argument('--gate_l1', type=float, default=0.01,
                        help='L1 penalty weight on gate (push toward 0)')
    parser.add_argument('--hard_neg_ratio', type=float, default=0.0,
                        help='Fraction of negatives sampled from historical neighbors (0=off)')
    parser.add_argument('--adaptive_alpha', action='store_true',
                        help='Per-source-node adaptive hard_neg_ratio: alpha(s) = hard_neg_ratio * (1 - node_rec_rate(s))')
    parser.add_argument('--warmup_ratio', type=float, default=0.8,
                        help='Fraction of training edges used for warmup (no gradient)')
    args = parser.parse_args()

    train_and_evaluate(args)
