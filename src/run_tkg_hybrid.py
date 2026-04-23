"""
TKG Link Prediction: Hybrid Decomposable + Sparse Pair Scoring

Architecture:
  - Decomposable score: per-entity features (degree, recency) + relation embedding
    → dot product → O(N) via matrix multiply
  - Sparse pair score: triple_count, pair_count, reverse_count
    → only non-zero for observed pairs → O(K << N)
  - Total score = decomposable + pair → fast for all-entity evaluation

This avoids the entity embedding problem (47K entities, too few training samples)
by using handcrafted features as the main signal.
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
import argparse
from collections import defaultdict


def compute_tie_aware_mrr(scores):
    """Match TGB's optimistic/pessimistic tie handling for a single query."""
    scores = np.asarray(scores, dtype=np.float32)
    pos_score = scores[0]
    neg_scores = scores[1:]
    optimistic_rank = float((neg_scores > pos_score).sum())
    pessimistic_rank = float((neg_scores >= pos_score).sum())
    return 1.0 / (0.5 * (optimistic_rank + pessimistic_rank) + 1.0)


def get_safe_result_path(base_dir, stem):
    """Avoid clobbering previous result files from earlier runs."""
    candidate = os.path.join(base_dir, f"{stem}.json")
    if not os.path.exists(candidate):
        return candidate
    suffix = 2
    while True:
        candidate = os.path.join(base_dir, f"{stem}_run{suffix}.json")
        if not os.path.exists(candidate):
            return candidate
        suffix += 1


def load_cached_state(cache_path):
    try:
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError, pickle.UnpicklingError, AttributeError, ValueError) as exc:
        print(f"  Cache at {cache_path} is unreadable ({exc.__class__.__name__}: {exc}). Rebuilding warmup state.")
        return None


def atomic_pickle_dump(obj, cache_path):
    tmp_path = f"{cache_path}.tmp.{os.getpid()}.{time.time_ns()}"
    try:
        with open(tmp_path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, cache_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def validate_args(args):
    """Fail fast on settings that otherwise crash later or produce NaNs."""
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1.")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1.")
    if args.n_neg < 1:
        raise ValueError("--n_neg must be >= 1.")


class HybridTKGScorer(nn.Module):
    """Hybrid scorer: decomposable entity features + sparse pair features.

    Score(src, rel, dst) = dst_score(dst, rel) + pair_bonus(src, rel, dst)
    where:
      dst_score = MLP(dst_features) conditioned on relation
      pair_bonus = w_triple * log1p(triple_count) + w_pair * log1p(pair_count)
                   + w_reverse * log1p(reverse_count)
    """

    def __init__(self, num_relations, entity_feat_dim, pair_feat_dim=4,
                 rel_dim=32, hidden_dim=64, dropout=0.1):
        super().__init__()

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

        # Source feature scorer: src_feats + rel_emb → query bias
        self.src_scorer = nn.Sequential(
            nn.Linear(entity_feat_dim + rel_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, src_feats, dst_feats, rel_ids, pair_feats):
        """Score specific (src, rel, dst) triplets.

        Args:
            src_feats: (B, entity_feat_dim)
            dst_feats: (B, entity_feat_dim)
            rel_ids: (B,)
            pair_feats: (B, pair_feat_dim)
        Returns:
            scores: (B,)
        """
        rel = self.rel_emb(rel_ids)  # (B, rel_dim)

        # Destination score
        dst_input = torch.cat([dst_feats, rel], dim=-1)
        dst_score = self.entity_scorer(dst_input).squeeze(-1)  # (B,)

        # Source bias
        src_input = torch.cat([src_feats, rel], dim=-1)
        src_score = self.src_scorer(src_input).squeeze(-1)  # (B,)

        # Pair score
        pair_input = torch.cat([pair_feats, rel], dim=-1)
        pair_score = self.pair_scorer(pair_input).squeeze(-1)  # (B,)

        return dst_score + src_score + pair_score


class IncrementalState:
    """Track graph state incrementally for feature computation."""

    def __init__(self, num_nodes, ema_decay=0.0):
        self.num_nodes = num_nodes
        self.ema_decay = ema_decay  # 0 = raw counts (default), >0 = EMA smoothing
        # Per-node
        self.out_degree = np.zeros(num_nodes, dtype=np.int32)
        self.in_degree = np.zeros(num_nodes, dtype=np.int32)
        self.last_time_src = np.full(num_nodes, -1e9)
        self.last_time_dst = np.full(num_nodes, -1e9)
        self.n_relations = np.zeros(num_nodes, dtype=np.int32)
        self.n_neighbors = np.zeros(num_nodes, dtype=np.int32)
        self._rel_sets = [set() for _ in range(num_nodes)]
        self._nbr_sets = [set() for _ in range(num_nodes)]

        # Per-pair (sparse) — indexed by source for fast lookup
        self.triple_count = defaultdict(int)   # (src, dst, rel) → count
        self.pair_count = defaultdict(int)      # (src, dst) → count
        self.rel_dst_count = defaultdict(int)   # (rel, dst) → count
        # EMA versions (only used when ema_decay > 0)
        self.triple_ema = defaultdict(float)
        self.pair_ema = defaultdict(float)
        self.rel_dst_ema = defaultdict(float)
        # Source-indexed for O(K) per-query pair scoring
        self.src_dsts = defaultdict(set)       # src → {dst, ...}
        self.src_rel_dsts = defaultdict(set)   # (src, rel) → {dst, ...}
        # Per-pair last interaction time (for collision-aware sampling)
        self.pair_last_time = {}               # (src, dst) → last_t

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
        self.pair_last_time[(src, dst)] = t
        if getattr(self, 'ema_decay', 0) > 0:
            alpha = self.ema_decay
            self.triple_ema[(src, dst, rel)] = (1-alpha) * self.triple_ema[(src, dst, rel)] + alpha
            self.pair_ema[(src, dst)] = (1-alpha) * self.pair_ema[(src, dst)] + alpha
            self.rel_dst_ema[(rel, dst)] = (1-alpha) * self.rel_dst_ema[(rel, dst)] + alpha
        self.src_dsts[src].add(dst)
        self.src_rel_dsts[(src, rel)].add(dst)

        self.current_time = t
        self.num_edges += 1

    def get_entity_features(self, node, t):
        """6-dim per-entity features."""
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
        """Vectorized 6-dim features for ALL entities."""
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

    def get_pair_features(self, src, dst, rel, t=None):
        """4-dim (or 5-dim if Trick B on) sparse pair features.
        Trick B: adds log(t - pair_last_time(s,d)) as 5th feature, -1 if unseen."""
        use_recency = getattr(self, 'pair_recency_feat', False) and t is not None
        if getattr(self, 'ema_decay', 0) > 0:
            base = [
                self.triple_ema.get((src, dst, rel), 0.0),
                self.pair_ema.get((src, dst), 0.0),
                self.pair_ema.get((dst, src), 0.0),
                self.rel_dst_ema.get((rel, dst), 0.0),
            ]
        else:
            base = [
                np.log1p(self.triple_count.get((src, dst, rel), 0)),
                np.log1p(self.pair_count.get((src, dst), 0)),
                np.log1p(self.pair_count.get((dst, src), 0)),
                np.log1p(self.rel_dst_count.get((rel, dst), 0)),
            ]
        if use_recency:
            last_t = self.pair_last_time.get((src, dst))
            sd_rec = np.log1p(max(0.0, t - last_t)) if last_t is not None else -1.0
            base.append(sd_rec)
        return np.array(base, dtype=np.float32)

    def get_pair_features_all_dsts(self, src, rel, num_nodes):
        """Get pair features for (src, rel, *) for ALL destination candidates.
        Returns sparse representation efficiently.

        Returns: (num_nodes, 4) array
        """
        feats = np.zeros((num_nodes, 4), dtype=np.float32)

        # Feature 3: rel_dst_count (independent of src)
        for (r, d), cnt in self.rel_dst_count.items():
            if r == rel and d < num_nodes:
                feats[d, 3] = np.log1p(cnt)

        # Features 0-2: depend on src
        # triple_count[(src, dst, rel)]
        for (s, d, r), cnt in self.triple_count.items():
            if s == src and r == rel and d < num_nodes:
                feats[d, 0] = np.log1p(cnt)

        # pair_count[(src, dst)]
        for (s, d), cnt in self.pair_count.items():
            if s == src and d < num_nodes:
                feats[d, 1] = np.log1p(cnt)

        # reverse: pair_count[(dst, src)]
        for (s, d), cnt in self.pair_count.items():
            if d == src and s < num_nodes:
                feats[s, 2] = np.log1p(cnt)

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
    try:
        dataset = LinkPropPredDataset(name=args.dataset, root="datasets")
        data = dataset.full_data
    except (OSError, IOError, ConnectionError, RuntimeError) as e:
        print(f"  TGB loader failed ({e.__class__.__name__}: {e}), trying direct pkl load...")
        import pandas as pd
        ds_dir = os.path.join("datasets", args.dataset.replace("-", "_"))
        pkl_path = os.path.join(ds_dir, f"ml_{args.dataset}.pkl")
        df = pd.read_pickle(pkl_path)
        data = {
            'sources': df['u'].values.astype(np.int64) if 'u' in df else df.iloc[:,0].values.astype(np.int64),
            'destinations': df['i'].values.astype(np.int64) if 'i' in df else df.iloc[:,1].values.astype(np.int64),
            'timestamps': df['ts'].values.astype(np.float64) if 'ts' in df else df.iloc[:,2].values.astype(np.float64),
        }
        if 'edge_type' in df:
            data['edge_type'] = df['edge_type'].values.astype(np.int64)
        # Note: 'label' column in TGB pkls is edge weight, not relation type
        # Build masks: 70/15/15 chronological split
        n = len(data['sources'])
        train_end_idx = int(n * 0.70)
        val_end_idx = int(n * 0.85)
        class _FakeDataset:
            def __init__(self, ns_dir, ds_name):
                self._ns_dir = ns_dir
                self._ds_name = ds_name
                self.negative_sampler = None
            def load_val_ns(self):
                self._load_ns('val')
            def load_test_ns(self):
                self._load_ns('test')
            def _load_ns(self, split):
                # NOTE: hist_rnd strategy is correct for tgbl-* LP datasets only.
                # TKG/THG datasets use different neg samplers but should not hit this fallback.
                from tgb.linkproppred.negative_sampler import NegativeEdgeSampler
                ns_pkl = os.path.join(self._ns_dir, f"{self._ds_name}_{split}_ns_v2.pkl")
                if os.path.exists(ns_pkl):
                    self.negative_sampler = NegativeEdgeSampler(dataset_name=self._ds_name, strategy='hist_rnd')
                    with open(ns_pkl, 'rb') as f:
                        self.negative_sampler.eval_set = {split: pickle.load(f)}
                else:
                    print(f"  WARNING: {ns_pkl} not found, eval may fail")
        dataset = _FakeDataset(ds_dir, args.dataset)
        dataset.train_mask = np.zeros(n, dtype=bool); dataset.train_mask[:train_end_idx] = True
        dataset.val_mask = np.zeros(n, dtype=bool); dataset.val_mask[train_end_idx:val_end_idx] = True
        dataset.test_mask = np.zeros(n, dtype=bool); dataset.test_mask[val_end_idx:] = True
        # Load negative samples if available (pickle already imported at top)
        for split in ['val', 'test']:
            ns_path = os.path.join(ds_dir, f"{args.dataset.replace('-','_')}_{split}_ns.npy")
            if os.path.exists(ns_path):
                setattr(dataset, f'{split}_ns', np.load(ns_path))
        dataset.full_data = data
        print(f"  Loaded from {pkl_path}: {n:,} edges")
    num_nodes = int(max(data['sources'].max(), data['destinations'].max())) + 1
    edge_types = data.get('edge_type', np.full(len(data['sources']), 0))
    num_relations = int(edge_types.max()) + 1

    sources = data['sources']
    destinations = data['destinations']
    timestamps = data['timestamps']
    ets = edge_types.astype(np.int64)

    total = len(sources)

    # Check if mask matches prefix (LP/THG are fine, TKG has inverse edges)
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask
    train_indices = np.where(train_mask)[0]
    is_prefix = np.array_equal(train_indices, np.arange(len(train_indices)))

    if is_prefix:
        # LP and THG: mask == prefix, use simple slicing
        train_end = int(train_mask.sum())
        val_end = train_end + int(val_mask.sum())
    else:
        # TKG: inverse edges break ordering. Reorder by mask.
        print(f"  WARNING: train_mask != prefix (inverse edges detected). Reordering by mask.")
        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
        test_idx = np.where(test_mask)[0]
        # Sort each split by timestamp for correct temporal ordering
        train_idx = train_idx[np.argsort(timestamps[train_idx], kind='stable')]
        val_idx = val_idx[np.argsort(timestamps[val_idx], kind='stable')]
        test_idx = test_idx[np.argsort(timestamps[test_idx], kind='stable')]
        reorder = np.concatenate([train_idx, val_idx, test_idx])
        sources = sources[reorder]
        destinations = destinations[reorder]
        timestamps = timestamps[reorder]
        ets = ets[reorder]
        # Now prefix slicing is correct AND each split is chronological
        train_end = len(train_idx)
        val_end = train_end + len(val_idx)
        # Update data dict so downstream (neg sampler etc) sees reordered data
        data['sources'] = sources
        data['destinations'] = destinations
        data['timestamps'] = timestamps
        if 'edge_type' in data:
            data['edge_type'] = ets

    print(f"  Nodes: {num_nodes}, Relations: {num_relations}, Edges: {total}")
    print(f"  Train: {train_end}, Val: {val_end-train_end}, Test: {total-val_end}")
    print(f"  Prefix-safe: {is_prefix}")

    # Pre-load future (val+test) pairs for oracle-filtered hard negatives
    future_pairs = None
    if getattr(args, 'filtered_hardneg', False):
        future_pairs = set()
        for i in range(train_end, total):
            future_pairs.add((int(sources[i]), int(destinations[i])))
        print(f"  [FILTERED_HARDNEG] Loaded {len(future_pairs)} future (val+test) pairs for oracle filtering")

    # 2. Build state (with cache)
    state = IncrementalState(num_nodes, ema_decay=args.ema_decay)
    warmup_end = train_end * 4 // 5

    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets")
    os.makedirs(cache_dir, exist_ok=True)
    ema_tag = f"_ema{args.ema_decay}" if args.ema_decay > 0 else ""
    reorder_tag = "_reordered" if not is_prefix else ""
    cache_key = f"{args.dataset}_warmup{warmup_end}_n{num_nodes}{ema_tag}{reorder_tag}_seed{args.seed}"
    cache_path = os.path.join(cache_dir, f"{cache_key}_state.pkl")

    if os.path.exists(cache_path):
        print(f"\nLoading cached warmup state from {cache_path}...")
        t0 = time.time()
        cached_state = load_cached_state(cache_path)
        if cached_state is not None:
            state = cached_state
            # Backward compat: add pair_last_time if missing from old cache
            # Use plain dict (not defaultdict with lambda) so state is pickle-safe
            # for later snapshot at line ~801. get_pair_features uses .get() to handle missing.
            if not hasattr(state, 'pair_last_time'):
                print("  (Rebuilding pair_last_time from warmup — old cache format)")
                state.pair_last_time = {}
                for i in range(warmup_end):
                    s_tmp, d_tmp, t_tmp = int(sources[i]), int(destinations[i]), float(timestamps[i])
                    state.pair_last_time[(s_tmp, d_tmp)] = t_tmp
            print(f"  Loaded in {time.time()-t0:.1f}s")
        else:
            os.makedirs(cache_dir, exist_ok=True)
            print(f"\nWarming up on first {warmup_end} edges...")
            t0 = time.time()
            for i in range(warmup_end):
                state.update(int(sources[i]), int(destinations[i]),
                             float(timestamps[i]), int(ets[i]))
            print(f"  Done in {time.time()-t0:.1f}s")
            print(f"  Saving state cache to {cache_path}...")
            atomic_pickle_dump(state, cache_path)
            print(f"  Saved.")
    else:
        print(f"\nWarming up on first {warmup_end} edges...")
        t0 = time.time()
        for i in range(warmup_end):
            state.update(int(sources[i]), int(destinations[i]),
                         float(timestamps[i]), int(ets[i]))
        print(f"  Done in {time.time()-t0:.1f}s")
        # Save cache
        print(f"  Saving state cache to {cache_path}...")
        atomic_pickle_dump(state, cache_path)
        print(f"  Saved.")

    # Type-constrained negative candidates
    type_to_dsts = defaultdict(set)
    for i in range(train_end):
        type_to_dsts[int(ets[i])].add(int(destinations[i]))
    type_to_dsts_arr = {et: np.array(list(d)) for et, d in type_to_dsts.items()}

    # Collision-aware hard negative setup
    collision_threshold = None
    collision_decay = 0.0
    collision_filtered_count = 0
    collision_total_hn_count = 0
    collision_fallback_count = 0
    if getattr(args, 'collision_aware', False) and getattr(args, 'hard_neg_ratio', 0.0) > 0:
        print("\n  [COLLISION_AWARE] Computing decay and threshold from warmup data...")
        # Compute decay = median inter-event time from warmup timestamps
        ts_warmup = timestamps[:warmup_end].astype(np.float64)
        if args.collision_decay > 0:
            collision_decay = args.collision_decay
            print(f"    Using user-specified decay = {collision_decay:.2f}")
        else:
            # Per-pair inter-event times: use a sample of pairs
            pair_times = defaultdict(list)
            for i in range(warmup_end):
                pair_times[(int(sources[i]), int(destinations[i]))].append(float(timestamps[i]))
            iets = []
            for pair_key, times_list in pair_times.items():
                if len(times_list) >= 2:
                    times_sorted = sorted(times_list)
                    for j in range(1, len(times_sorted)):
                        dt = times_sorted[j] - times_sorted[j-1]
                        if dt > 0:
                            iets.append(dt)
            if len(iets) > 0:
                collision_decay = float(np.median(iets))
            else:
                # Fallback: use global median inter-event time
                dts = np.diff(ts_warmup)
                dts = dts[dts > 0]
                collision_decay = float(np.median(dts)) if len(dts) > 0 else 1.0
            print(f"    Auto decay (median pair inter-event time) = {collision_decay:.4f}")

        # Sample collision scores from existing hist_neighbors to set threshold
        # Use a random sample of training edges to estimate score distribution
        n_sample = min(10000, n_train_edges := (train_end - warmup_end))
        sample_indices = np.random.choice(range(warmup_end, train_end), size=n_sample, replace=False)
        sample_scores = []
        for si in sample_indices:
            s_i = int(sources[si])
            t_i = float(timestamps[si])
            d_i = int(destinations[si])
            neighbors = list(state.src_dsts.get(s_i, set()) - {d_i})
            if len(neighbors) > 0:
                # Sample a few neighbors, compute their scores
                n_check = min(5, len(neighbors))
                for _ in range(n_check):
                    nd_i = neighbors[np.random.randint(len(neighbors))]
                    # Inline collision score to avoid pickle/class version issues
                    _pc = state.pair_count.get((s_i, nd_i), 0)
                    _lt = state.pair_last_time.get((s_i, nd_i), None) if hasattr(state, 'pair_last_time') else None
                    if _pc > 0 and _lt is not None and collision_decay > 0:
                        sc = _pc * np.exp(-(t_i - _lt) / collision_decay)
                    else:
                        sc = 0.0
                    if sc > 0:
                        sample_scores.append(sc)

        # Debug: check why no scores found
        if len(sample_scores) == 0:
            n_with_neighbors = sum(1 for si in sample_indices[:100]
                                   if len(state.src_dsts.get(int(sources[si]), set())) > 1)
            sample_s = int(sources[sample_indices[0]])
            sample_nbrs = list(state.src_dsts.get(sample_s, set()))[:3]
            print(f"    DEBUG: {n_with_neighbors}/100 samples have neighbors")
            print(f"    DEBUG: sample src={sample_s}, neighbors={sample_nbrs}")
            for nb in sample_nbrs:
                pc = state.pair_count.get((sample_s, nb), 0)
                lt = state.pair_last_time.get((sample_s, nb), 'MISS')
                print(f"      pair_count({sample_s},{nb})={pc}, last_time={lt}")
            print(f"    DEBUG: hasattr pair_last_time={hasattr(state, 'pair_last_time')}, "
                  f"type={type(getattr(state, 'pair_last_time', None))}, "
                  f"len={len(getattr(state, 'pair_last_time', {}))}")
        if len(sample_scores) > 0:
            collision_threshold = float(np.percentile(sample_scores, args.collision_percentile))
            print(f"    Collision scores: n={len(sample_scores)}, "
                  f"median={np.median(sample_scores):.4f}, "
                  f"p75={np.percentile(sample_scores, 75):.4f}, "
                  f"p90={np.percentile(sample_scores, 90):.4f}, "
                  f"p95={np.percentile(sample_scores, 95):.4f}")
            print(f"    Threshold (p{args.collision_percentile:.0f}) = {collision_threshold:.4f}")
        else:
            print("    WARNING: No non-zero collision scores found. Collision-aware filtering disabled.")

    # 3. Collect training data
    n_train = train_end - warmup_end
    if n_train < 1:
        raise ValueError(
            f"{args.dataset} has no trainable edges after warmup (train_end={train_end}, warmup_end={warmup_end})."
        )
    n_neg = args.n_neg
    print(f"\nCollecting {n_train} trainable edges...")
    t0 = time.time()

    entity_feat_dim = 6
    pair_feat_dim = 5 if getattr(args, 'pair_recency_feat', False) else 4
    state.pair_recency_feat = getattr(args, 'pair_recency_feat', False)

    train_src_feats = np.zeros((n_train, entity_feat_dim), dtype=np.float32)
    train_dst_feats = np.zeros((n_train, entity_feat_dim), dtype=np.float32)
    train_pair_feats = np.zeros((n_train, pair_feat_dim), dtype=np.float32)
    train_neg_dst_feats = np.zeros((n_train, n_neg, entity_feat_dim), dtype=np.float32)
    train_neg_pair_feats = np.zeros((n_train, n_neg, pair_feat_dim), dtype=np.float32)
    train_rels = np.zeros(n_train, dtype=np.int64)
    filtered_collision_count = 0
    total_hard_neg_count = 0

    for i in range(n_train):
        idx = warmup_end + i
        s = int(sources[idx])
        d = int(destinations[idx])
        t_i = float(timestamps[idx])
        r = int(ets[idx])

        train_src_feats[i] = state.get_entity_features(s, t_i)
        train_dst_feats[i] = state.get_entity_features(d, t_i)
        train_pair_feats[i] = state.get_pair_features(s, d, r, t_i)
        train_rels[i] = r

        # Negatives: mix of random and hard (from historical neighbors of s)
        candidates = type_to_dsts_arr.get(r, None)
        hard_neg_ratio = getattr(args, 'hard_neg_ratio', 0.0)
        n_hard = int(n_neg * hard_neg_ratio)
        hist_neighbors = list(state.src_dsts.get(s, set()) - {d}) if hard_neg_ratio > 0 else []

        for k in range(n_neg):
            if k < n_hard and len(hist_neighbors) > 0:
                # Hard negative: sample from s's historical neighbors (excluding d)
                total_hard_neg_count += 1
                nd = hist_neighbors[np.random.randint(len(hist_neighbors))]
                # Oracle filter: skip if (s, nd) appears in future set
                if future_pairs is not None and (s, nd) in future_pairs:
                    filtered_collision_count += 1
                    for _retry in range(10):
                        nd = hist_neighbors[np.random.randint(len(hist_neighbors))]
                        if (s, nd) not in future_pairs:
                            break
                    else:
                        # All retries collided — fallback to random
                        nd = np.random.randint(num_nodes)
                # Non-oracle collision-aware filter: skip high recurrence+recency pairs
                elif collision_threshold is not None:
                    _pc = state.pair_count.get((s, nd), 0)
                    _lt = state.pair_last_time.get((s, nd), None) if hasattr(state, 'pair_last_time') else None
                    cs = _pc * np.exp(-(t_i - _lt) / collision_decay) if (_pc > 0 and _lt is not None and collision_decay > 0) else 0.0
                    if cs > collision_threshold:
                        collision_filtered_count += 1
                        collision_total_hn_count += 1
                        found_safe = False
                        for _retry in range(10):
                            nd = hist_neighbors[np.random.randint(len(hist_neighbors))]
                            _pc2 = state.pair_count.get((s, nd), 0)
                            _lt2 = state.pair_last_time.get((s, nd), None) if hasattr(state, 'pair_last_time') else None
                            cs2 = _pc2 * np.exp(-(t_i - _lt2) / collision_decay) if (_pc2 > 0 and _lt2 is not None and collision_decay > 0) else 0.0
                            if cs2 <= collision_threshold:
                                found_safe = True
                                break
                        if not found_safe:
                            # All retries above threshold — fallback to random
                            nd = np.random.randint(num_nodes)
                            collision_fallback_count += 1
                    else:
                        collision_total_hn_count += 1
            elif candidates is not None and len(candidates) > 1:
                nd = candidates[np.random.randint(len(candidates))]
                while nd == d:
                    nd = candidates[np.random.randint(len(candidates))]
            else:
                nd = np.random.randint(num_nodes)
            nd = int(nd)
            train_neg_dst_feats[i, k] = state.get_entity_features(nd, t_i)
            train_neg_pair_feats[i, k] = state.get_pair_features(s, nd, r, t_i)

        state.update(s, d, t_i, r)

        if (i + 1) % 50000 == 0:
            print(f"    {i+1}/{n_train} ({time.time()-t0:.1f}s)")

    print(f"  Done in {time.time()-t0:.1f}s")
    if future_pairs is not None and total_hard_neg_count > 0:
        print(f"  [FILTERED_HARDNEG] Collisions filtered: {filtered_collision_count}/{total_hard_neg_count} "
              f"({100*filtered_collision_count/total_hard_neg_count:.1f}%)")
    if collision_threshold is not None and collision_total_hn_count > 0:
        print(f"  [COLLISION_AWARE] Filtered: {collision_filtered_count}/{collision_total_hn_count} "
              f"({100*collision_filtered_count/collision_total_hn_count:.1f}%), "
              f"fallback_to_random: {collision_fallback_count}, "
              f"threshold: {collision_threshold:.4f}, decay: {collision_decay:.4f}")

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

    # Relation-aware mode: zero out relation-agnostic pair features (indices 1,2)
    # With 5d features: [0:triple, 1:pair, 2:reverse, 3:rel_dst, 4:cross_rel]
    # cross_rel (idx 4) captures "interacted before with different relation" — keep it
    if args.rel_aware_pairs:
        print("  [REL_AWARE] Zeroing pair_count features (keeping triple_count, rel_dst_count, cross_rel_count)")
        train_pair_feats[:, 1] = 0
        train_pair_feats[:, 2] = 0
        train_neg_pair_feats[:, :, 1] = 0
        train_neg_pair_feats[:, :, 2] = 0

    # 4. Model
    model = HybridTKGScorer(
        num_relations, entity_feat_dim, pair_feat_dim,
        rel_dim=args.rel_dim, hidden_dim=args.hidden_dim, dropout=args.dropout
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # Tensors
    t_src_f = torch.FloatTensor(train_src_feats).to(device)
    t_dst_f = torch.FloatTensor(train_dst_feats).to(device)
    t_pair_f = torch.FloatTensor(train_pair_feats).to(device)
    t_neg_dst_f = torch.FloatTensor(train_neg_dst_feats).to(device)
    t_neg_pair_f = torch.FloatTensor(train_neg_pair_feats).to(device)
    t_rels = torch.LongTensor(train_rels).to(device)

    # Ablation: zero out features
    if args.ablation == 'entity_only':
        t_pair_f.zero_()
        t_neg_pair_f.zero_()
        print("  [ABLATION] entity_only: pair features zeroed")
    elif args.ablation == 'pair_only':
        t_src_f.zero_()
        t_dst_f.zero_()
        t_neg_dst_f.zero_()
        print("  [ABLATION] pair_only: entity features zeroed")

    # 5. Train with checkpoint saving for post-hoc evaluation
    print(f"\nTraining for {args.epochs} epochs...")
    checkpoints = {}  # epoch -> state_dict (on CPU)
    best_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    patience = args.patience

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        total_loss = 0
        n_batches = 0

        for i in range(0, n_train, args.batch_size):
            idx = perm[i:i + args.batch_size]
            b_src_f = t_src_f[idx]
            b_dst_f = t_dst_f[idx]
            b_pair_f = t_pair_f[idx]
            b_rels = t_rels[idx]

            # Positive scores
            pos_scores = model(b_src_f, b_dst_f, b_rels, b_pair_f)

            # Negative scores
            neg_scores_list = []
            for k in range(n_neg):
                ns = model(b_src_f, t_neg_dst_f[idx, k], b_rels,
                          t_neg_pair_f[idx, k])
                neg_scores_list.append(ns)
            neg_scores = torch.stack(neg_scores_list, dim=1)  # (B, n_neg)

            # Loss computation
            if args.loss == 'bpr':
                # BPR: pairwise log-sigmoid
                diff = pos_scores.unsqueeze(1) - neg_scores
                loss = -F.logsigmoid(diff).mean()
            elif args.loss == 'ce':
                # Cross-Entropy: pos should rank first among {pos, neg1..negK}
                all_scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)  # (B, 1+K)
                targets = torch.zeros(len(idx), dtype=torch.long, device=device)  # pos is index 0
                loss = F.cross_entropy(all_scores, targets)
            elif args.loss == 'margin':
                # Margin-Hinge (inspired by NAVIS): max(0, δ - (pos - neg))
                margin = 0.1
                diff = pos_scores.unsqueeze(1) - neg_scores
                loss = torch.clamp(margin - diff, min=0).mean()
            else:
                raise ValueError(f"Unknown loss: {args.loss}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)

        # Save checkpoint every epoch (model is tiny, ~50KB per checkpoint)
        checkpoints[epoch + 1] = {
            'state_dict': {k: v.cpu().clone() for k, v in model.state_dict().items()},
            'loss': avg_loss,
        }

        # Track best loss for logging
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f} "
                  f"(best_loss: {best_loss:.4f} @ ep{best_epoch}, {len(checkpoints)} ckpts saved)")

        if patience_counter >= patience and epoch >= 10:
            print(f"  Early stopping at epoch {epoch+1} (no loss improvement for {patience} epochs)")
            break

    print(f"  Training done. {len(checkpoints)} checkpoints saved.")

    # 6. Evaluate: GET-style checkpoint selection
    eval_start_time = time.time()
    # For each checkpoint (every eval_interval epochs):
    #   snapshot state → process val edges → compute val MRR
    #   if best val → also compute test MRR → restore state
    # Report best test MRR among epochs that achieved best val MRR
    # Determine eval interval based on graph size
    eval_interval = 5
    if num_nodes > 500000:
        eval_interval = 10
    elif num_nodes < 50000:
        eval_interval = 1

    eval_epochs = sorted([ep for ep in checkpoints.keys()
                          if ep % eval_interval == 0 or ep == 1 or ep == max(checkpoints.keys())])
    print(f"\n{'='*60}")
    print(f"GET-style evaluation: {len(eval_epochs)} checkpoints, interval={eval_interval}")

    # Snapshot state before any eval
    state_snapshot = pickle.dumps(state)

    dataset.load_val_ns()
    val_neg_sampler = dataset.negative_sampler
    val_eval_set = val_neg_sampler.eval_set.get('val', {})

    dataset.load_test_ns()
    test_neg_sampler = dataset.negative_sampler
    test_eval_set = test_neg_sampler.eval_set.get('test', {})
    is_linkpred = args.dataset.startswith('tgbl-')
    lp_eval_uses_edge_type = False
    if is_linkpred:
        for eval_set in (val_eval_set, test_eval_set):
            if eval_set:
                lp_eval_uses_edge_type = len(next(iter(eval_set))) == 4
                break

    best_val_mrr = -1.0
    best_test_mrr = -1.0
    best_epoch_val = -1
    best_epoch_test = -1
    all_results = []

    def _eval_split(split_srcs, split_dsts, split_ts, split_ets, neg_sam, ev_set, split_mode, max_queries=999999999):
        """Evaluate MRR on a split (val or test) using current model and state.

        For LP (tgbl-*): iterate over split edges, score those in eval_set.
        For TKG (tkgl-*/thgl-*): first process all split edges to update state,
            then iterate over eval_set queries (which may have different timestamps).
        """
        if len(split_srcs) == 0:
            return 0.0, 0
        mrr_list = []
        n_eval = 0

        if is_linkpred:
            # LP mode: interleave scoring and state updates
            all_ef = state.get_all_entity_features(float(split_ts[0]))
            all_ef_norm = (all_ef - ent_mean) / ent_std

            for vi in range(len(split_srcs)):
                vs, vd = int(split_srcs[vi]), int(split_dsts[vi])
                vt, ve = float(split_ts[vi]), int(split_ets[vi])
                if lp_eval_uses_edge_type:
                    vkey = (vs, vd, int(split_ts[vi]), ve)
                else:
                    vkey = (vs, vd, int(split_ts[vi]))
                if vkey not in ev_set:
                    state.update(vs, vd, vt, ve)
                    continue
                if n_eval >= max_queries:
                    state.update(vs, vd, vt, ve)
                    continue

                query_kwargs = {"split_mode": split_mode}
                if lp_eval_uses_edge_type:
                    query_kwargs["edge_type"] = np.array([ve])
                nb = neg_sam.query_batch(
                    np.array([vs]),
                    np.array([vd]),
                    np.array([split_ts[vi]]),
                    **query_kwargs,
                )
                neg_dsts_j = np.array(nb[0]).astype(int)

                if n_eval % 500 == 0:
                    all_ef = state.get_all_entity_features(vt)
                    all_ef_norm = (all_ef - ent_mean) / ent_std

                n_cand = len(neg_dsts_j) + 1
                cand_dsts = np.concatenate([[vd], neg_dsts_j])
                with torch.no_grad():
                    src_t = torch.FloatTensor(all_ef_norm[vs]).unsqueeze(0).expand(n_cand, -1).to(device)
                    dst_t = torch.FloatTensor(all_ef_norm[cand_dsts]).to(device)
                    if args.ablation == 'pair_only':
                        src_t = torch.zeros_like(src_t)
                        dst_t = torch.zeros_like(dst_t)
                    rel_t = torch.LongTensor(np.full(n_cand, ve)).to(device)
                    pf = np.zeros((n_cand, pair_feat_dim), dtype=np.float32)
                    for j, cd in enumerate(cand_dsts):
                        pf[j] = state.get_pair_features(vs, int(cd), ve, vt)
                    pf = (pf - pair_mean) / pair_std
                    if args.rel_aware_pairs:
                        pf[:, 1] = 0; pf[:, 2] = 0
                    if args.ablation == 'entity_only':
                        pf[:] = 0
                    scores = model(src_t, dst_t, rel_t, torch.FloatTensor(pf).to(device)).cpu().numpy()

                mrr_list.append(compute_tie_aware_mrr(scores))
                n_eval += 1
                state.update(vs, vd, vt, ve)
        else:
            # TKG mode: process ALL split edges first, then evaluate on eval_set queries
            for vi in range(len(split_srcs)):
                state.update(int(split_srcs[vi]), int(split_dsts[vi]),
                             float(split_ts[vi]), int(split_ets[vi]))

            all_ef = state.get_all_entity_features(float(split_ts[-1]))
            all_ef_norm = (all_ef - ent_mean) / ent_std

            # Build lookup: (ts, src, rel) → dst for fast positive lookup
            edge_lookup = {}
            for vi in range(len(split_srcs)):
                ekey = (int(split_ts[vi]), int(split_srcs[vi]), int(split_ets[vi]))
                edge_lookup[ekey] = int(split_dsts[vi])

            for query_key, neg_dsts_j in ev_set.items():
                if n_eval >= max_queries:
                    break
                q_ts, q_src, q_rel = int(query_key[0]), int(query_key[1]), int(query_key[2])
                neg_dsts_j = np.array(neg_dsts_j).astype(int).flatten()

                pos_dst = edge_lookup.get((q_ts, q_src, q_rel))
                if pos_dst is None:
                    continue

                n_cand = len(neg_dsts_j) + 1
                cand_dsts = np.concatenate([[pos_dst], neg_dsts_j])

                if n_eval % 500 == 0:
                    all_ef = state.get_all_entity_features(float(q_ts))
                    all_ef_norm = (all_ef - ent_mean) / ent_std

                with torch.no_grad():
                    src_t = torch.FloatTensor(all_ef_norm[q_src]).unsqueeze(0).expand(n_cand, -1).to(device)
                    dst_t = torch.FloatTensor(all_ef_norm[cand_dsts]).to(device)
                    if args.ablation == 'pair_only':
                        src_t = torch.zeros_like(src_t)
                        dst_t = torch.zeros_like(dst_t)
                    rel_t = torch.LongTensor(np.full(n_cand, q_rel)).to(device)
                    pf = np.zeros((n_cand, pair_feat_dim), dtype=np.float32)
                    for j, cd in enumerate(cand_dsts):
                        pf[j] = state.get_pair_features(q_src, int(cd), q_rel, float(q_ts))
                    pf = (pf - pair_mean) / pair_std
                    if args.rel_aware_pairs:
                        pf[:, 1] = 0; pf[:, 2] = 0
                    if args.ablation == 'entity_only':
                        pf[:] = 0
                    scores = model(src_t, dst_t, rel_t, torch.FloatTensor(pf).to(device)).cpu().numpy()

                mrr_list.append(compute_tie_aware_mrr(scores))
                n_eval += 1

        return np.mean(mrr_list) if mrr_list else 0.0, n_eval

    val_srcs, val_dsts = sources[train_end:val_end], destinations[train_end:val_end]
    val_ts_arr, val_ets_arr = timestamps[train_end:val_end], ets[train_end:val_end]
    test_srcs_arr, test_dsts_arr = sources[val_end:], destinations[val_end:]
    test_ts_arr, test_ets_arr = timestamps[val_end:], ets[val_end:]

    if is_linkpred:
        # LP: evaluate each checkpoint on val, select best val MRR
        for ep in eval_epochs:
            state = pickle.loads(state_snapshot)
            ckpt = checkpoints[ep]
            model.load_state_dict({k: v.to(device) for k, v in ckpt['state_dict'].items()})
            model.eval()

            val_mrr, n_val = _eval_split(val_srcs, val_dsts, val_ts_arr, val_ets_arr,
                                          val_neg_sampler, val_eval_set, 'val')

            if val_mrr > best_val_mrr:
                best_val_mrr = val_mrr
                best_epoch_val = ep

            all_results.append((ep, ckpt['loss'], val_mrr))
            print(f"  ep{ep}: loss={ckpt['loss']:.4f}, val_mrr={val_mrr:.4f}")

        print(f"\n  Best val: ep{best_epoch_val} val_mrr={best_val_mrr:.4f}")
    else:
        # TKG: val eval is complex (queries at future timestamps), use best-loss checkpoint
        best_epoch_val = min(checkpoints.keys(), key=lambda k: checkpoints[k]['loss'])
        print(f"  TKG mode: selecting best-loss checkpoint ep{best_epoch_val} "
              f"(loss={checkpoints[best_epoch_val]['loss']:.4f})")

    # Run test ONCE on the best-val checkpoint (no test leakage)
    print(f"  Running full test on ep{best_epoch_val}...")
    state = pickle.loads(state_snapshot)
    ckpt = checkpoints[best_epoch_val]
    model.load_state_dict({k: v.to(device) for k, v in ckpt['state_dict'].items()})
    model.eval()

    # Process val edges first (update state)
    for i in range(train_end, val_end):
        state.update(int(sources[i]), int(destinations[i]),
                     float(timestamps[i]), int(ets[i]))

    if is_linkpred:
        # LP: use _eval_split (processes edges chronologically, score before update)
        best_test_mrr, n_test = _eval_split(test_srcs_arr, test_dsts_arr, test_ts_arr, test_ets_arr,
                                             test_neg_sampler, test_eval_set, 'test')
    else:
        # TKG/THG: process test edges one by one (score BEFORE update, no future leakage)
        mrr_list = []
        t0_eval = time.time()
        if len(test_srcs_arr) > 0:
            all_ef = state.get_all_entity_features(float(test_ts_arr[0]))
            all_ef_norm = (all_ef - ent_mean) / ent_std
            recompute_ctr = 0

            for vi in range(len(test_srcs_arr)):
                vs, vd = int(test_srcs_arr[vi]), int(test_dsts_arr[vi])
                vt, ve = float(test_ts_arr[vi]), int(test_ets_arr[vi])
                vkey = (int(test_ts_arr[vi]), vs, ve)

                if vkey not in test_eval_set:
                    state.update(vs, vd, vt, ve)
                    recompute_ctr += 1
                    continue

                if recompute_ctr >= 500:
                    all_ef = state.get_all_entity_features(vt)
                    all_ef_norm = (all_ef - ent_mean) / ent_std
                    recompute_ctr = 0

                nb = test_neg_sampler.query_batch(
                    np.array([vs]), np.array([vd]),
                    np.array([test_ts_arr[vi]]), edge_type=np.array([ve]),
                    split_mode='test')
                neg_dsts_j = np.array(nb[0]).astype(int).flatten()

                n_cand = len(neg_dsts_j) + 1
                cand_dsts = np.concatenate([[vd], neg_dsts_j])

                with torch.no_grad():
                    src_t = torch.FloatTensor(all_ef_norm[vs]).unsqueeze(0).expand(n_cand, -1).to(device)
                    dst_t = torch.FloatTensor(all_ef_norm[cand_dsts]).to(device)
                    if args.ablation == 'pair_only':
                        src_t = torch.zeros_like(src_t)
                        dst_t = torch.zeros_like(dst_t)
                    rel_t = torch.LongTensor(np.full(n_cand, ve)).to(device)
                    pf = np.zeros((n_cand, pair_feat_dim), dtype=np.float32)
                    for j, cd in enumerate(cand_dsts):
                        pf[j] = state.get_pair_features(vs, int(cd), ve, vt)
                    pf = (pf - pair_mean) / pair_std
                    if args.rel_aware_pairs:
                        pf[:, 1] = 0; pf[:, 2] = 0
                    if args.ablation == 'entity_only':
                        pf[:] = 0
                    scores = model(src_t, dst_t, rel_t, torch.FloatTensor(pf).to(device)).cpu().numpy()

                mrr_list.append(compute_tie_aware_mrr(scores))

                state.update(vs, vd, vt, ve)
                recompute_ctr += 1

                if len(mrr_list) % 2000 == 0:
                    print(f"    {len(mrr_list)}/{len(test_eval_set)}, MRR: {np.mean(mrr_list):.4f} ({time.time()-t0_eval:.1f}s)")

        best_test_mrr = np.mean(mrr_list) if mrr_list else 0.0
        n_test = len(mrr_list)

    best_epoch_test = best_epoch_val
    print(f"  Test MRR: {best_test_mrr:.6f} ({n_test} queries)")

    final_mrr = best_test_mrr
    num_evaluated = n_test
    elapsed = time.time() - eval_start_time
    print(f"\n{'='*60}")
    print(f"RESULT: {args.dataset} hybrid MRR = {final_mrr:.6f}")
    print(f"  Best val ep{best_epoch_val} (val_mrr={best_val_mrr:.4f}), best test ep{best_epoch_test}")
    print(f"  Params: {n_params:,}")
    print(f"{'='*60}")

    # Build config tag for result tracking
    config_parts = []
    if hasattr(args, 'ablation') and args.ablation and args.ablation != 'none':
        config_parts.append(args.ablation)
    if hasattr(args, 'ema_decay') and args.ema_decay > 0:
        config_parts.append(f"ema{args.ema_decay}")
    if hasattr(args, 'rel_aware_pairs') and args.rel_aware_pairs:
        config_parts.append("relaware")
    if hasattr(args, 'rel_dim') and args.rel_dim != 32:
        config_parts.append(f"rd{args.rel_dim}")
    if hasattr(args, 'loss') and args.loss != 'bpr':
        config_parts.append(f"loss_{args.loss}")
    if hasattr(args, 'lr') and args.lr != 1e-3:
        config_parts.append(f"lr{args.lr}")
    if hasattr(args, 'n_neg') and args.n_neg != 10:
        config_parts.append(f"neg{args.n_neg}")
    config_tag = "_".join(config_parts) if config_parts else "default"

    result_dict = {
        'dataset': args.dataset,
        'model': 'hybrid',
        'config': config_tag,
        'mrr': float(final_mrr),
        'seed': args.seed,
        'n_params': n_params,
        'n_evaluated': num_evaluated,
        'eval_time': elapsed,
        'best_epoch': best_epoch_val,
    }
    os.makedirs("/data/chenlibin/TGB2/results", exist_ok=True)
    out_path = get_safe_result_path(
        "/data/chenlibin/TGB2/results",
        f"{args.dataset}_{config_tag}_seed{args.seed}",
    )
    with open(out_path, 'w') as f:
        json.dump(result_dict, f, indent=2)
    print(f"Saved to {out_path}")

    return final_mrr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='tkgl-smallpedia')
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
    parser.add_argument('--ablation', type=str, default='none',
                        choices=['none', 'entity_only', 'pair_only'],
                        help='Ablation: zero out pair or entity features')
    parser.add_argument('--rel_aware_pairs', action='store_true',
                        help='Zero out relation-agnostic pair_count features (keep only triple_count and rel_dst_count)')
    parser.add_argument('--ema_decay', type=float, default=0.0,
                        help='EMA decay for pair features (0=raw counts, >0=exponential smoothing)')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--hard_neg_ratio', type=float, default=0.0,
                        help='Fraction of negatives from historical neighbors (0=random)')
    parser.add_argument('--filtered_hardneg', action='store_true',
                        help='Oracle filter: skip hard negatives that appear in future test set')
    parser.add_argument('--collision_aware', action='store_true',
                        help='Non-oracle collision-aware hard neg: skip candidates with high recurrence+recency')
    parser.add_argument('--collision_percentile', type=float, default=75.0,
                        help='Percentile threshold for collision score filtering (default: 75)')
    parser.add_argument('--collision_decay', type=float, default=0.0,
                        help='Decay for recency weight (0=auto from median inter-event time)')
    parser.add_argument('--loss', type=str, default='bpr', choices=['bpr', 'ce', 'margin'],
                        help='Loss function: bpr (default), ce (cross-entropy), margin (hinge)')
    parser.add_argument('--auto', action='store_true',
                        help='Auto-adapt: compute dataset stats and select best config')
    parser.add_argument('--pair_recency_feat', action='store_true',
                        help='Trick B: add log(t - pair_last_time(s,d)) as 5th pair feature (default off for backward compat)')
    args = parser.parse_args()

    if args.auto:
        # Auto-adapt: compute dataset statistics and select config
        print("="*60)
        print("AUTO-ADAPT: Computing dataset statistics...")
        dataset_auto = LinkPropPredDataset(name=args.dataset, root="datasets")
        data_auto = dataset_auto.full_data
        src = data_auto['sources']
        dst = data_auto['destinations']
        et = data_auto.get('edge_type', np.zeros(len(src), dtype=int))

        # Handle TKG inverse-edge reorder (same logic as train_and_evaluate)
        train_mask_auto = dataset_auto.train_mask
        val_mask_auto = dataset_auto.val_mask
        train_idx_auto = np.where(train_mask_auto)[0]
        is_prefix_auto = np.array_equal(train_idx_auto, np.arange(len(train_idx_auto)))
        if is_prefix_auto:
            train_end = int(train_mask_auto.sum())
            val_end = train_end + int(val_mask_auto.sum())
        else:
            val_idx_auto = np.where(val_mask_auto)[0]
            test_idx_auto = np.where(dataset_auto.test_mask)[0]
            reorder_auto = np.concatenate([train_idx_auto, val_idx_auto, test_idx_auto])
            src = src[reorder_auto]
            dst = dst[reorder_auto]
            et = et[reorder_auto]
            train_end = len(train_idx_auto)
            val_end = train_end + len(val_idx_auto)
            print(f"  (TKG reorder applied for auto-adapt)")

        n_rel = int(et.max()) + 1

        # Pair recurrence rate
        train_pairs = set()
        train_triples = set()
        for i in range(train_end):
            train_pairs.add((int(src[i]), int(dst[i])))
            train_triples.add((int(src[i]), int(dst[i]), int(et[i])))
        # Use VAL set (not test!) to compute pair_rec — avoids test leakage
        n_val = val_end - train_end
        pair_rec = sum(1 for i in range(train_end, val_end)
                       if (int(src[i]), int(dst[i])) in train_pairs) / max(n_val, 1)
        triple_rec = sum(1 for i in range(train_end, val_end)
                         if (int(src[i]), int(dst[i]), int(et[i])) in train_triples) / max(n_val, 1)
        gap = pair_rec - triple_rec

        print(f"  n_rels={n_rel}, pair_rec={pair_rec:.3f}, triple_rec={triple_rec:.3f}, gap={gap:.3f}")

        # Rule 0: TKG datasets with many relations → lower lr + more negatives
        n_nodes_total = int(max(src.max(), dst.max())) + 1
        is_tkg = args.dataset.startswith('tkgl-')
        if is_tkg and n_rel > 5 and args.lr == 1e-3:
            args.lr = 1e-4
            args.n_neg = 50
            print(f"  AUTO: lr → 1e-4, n_neg → 50 (TKG with {n_rel} rels)")

        # Rule 1: Many relations → larger rel_dim
        if n_rel > 100 and args.rel_dim == 32:
            args.rel_dim = 128
            print(f"  AUTO: rel_dim → 128 (n_rels={n_rel} > 100)")

        # Rule 2: Large pair-triple gap → rel_aware_pairs
        if gap > 0.3 and not args.rel_aware_pairs:
            args.rel_aware_pairs = True
            print(f"  AUTO: rel_aware_pairs → True (gap={gap:.3f} > 0.3)")

        # Rule 3: Low pair_rec + high diversity → hard negatives
        unique_pairs_tmp = len(train_pairs)
        diversity_tmp = unique_pairs_tmp / max(train_end, 1)
        if pair_rec < 0.10 and args.hard_neg_ratio == 0.0:
            args.hard_neg_ratio = 0.2
            print(f"  AUTO: hard_neg_ratio → 0.2 (pair_rec={pair_rec:.3f} < 0.10)")

        # Rule 4: High pair_rec + enough nodes → EMA smoothing
        # Small graphs (<5K nodes) have fast-updating counts, EMA causes lag
        unique_pairs = len(train_pairs)
        diversity = unique_pairs / max(train_end, 1)
        n_nodes_total = int(max(src.max(), dst.max())) + 1
        if pair_rec > 0.30 and n_nodes_total > 5000 and args.ema_decay == 0.0:
            args.ema_decay = 0.1
            print(f"  AUTO: ema_decay → 0.1 (pair_rec={pair_rec:.3f} > 0.30, nodes={n_nodes_total:,} > 5K)")

        # Rule 5: Super-hub + historical neg eval → entity_only
        # When max_degree is very high, pair features collide with TGB historical negatives
        from collections import Counter
        src_deg = Counter(src[:train_end].tolist())
        max_deg = max(src_deg.values()) if src_deg else 0
        if max_deg > 20000 and pair_rec > 0.30 and args.ablation == 'none' and not args.dataset.startswith('thgl-') and not args.dataset.startswith('tkgl-'):
            # Only for LP datasets with high pair_rec + super-hub (historical neg collision)
            # Low pair_rec (like comment 0.037) means pair features are weak anyway, no collision
            args.ablation = 'entity_only'
            print(f"  AUTO: ablation → entity_only (max_degree={max_deg:,} > 20K, historical neg collision risk)")

        print(f"  Final config: rel_dim={args.rel_dim}, rel_aware={args.rel_aware_pairs}, "
              f"hard_neg={args.hard_neg_ratio}, filtered_hardneg={args.filtered_hardneg}, "
              f"collision_aware={args.collision_aware}, "
              f"ema={args.ema_decay}, ablation={args.ablation}")
        print("="*60)
        del dataset_auto, data_auto  # Free memory before training

    train_and_evaluate(args)
