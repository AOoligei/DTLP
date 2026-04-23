from collections import defaultdict

import numpy as np


class TemporalStatsBank:

    def __init__(self, num_nodes: int, all_dst_node_ids: np.ndarray, rho_ema_momentum: float = 0.95, seed: int = None):
        self.num_nodes = int(num_nodes)
        self.all_dst_node_ids = np.asarray(all_dst_node_ids, dtype=np.int64)
        self.rho_ema_momentum = float(rho_ema_momentum)
        self.seed = seed
        self.random_state = np.random.RandomState(self.seed) if self.seed is not None else np.random.RandomState()
        self.reset()

    def reset(self):
        self.pair_count = defaultdict(int)
        self.pair_last_time = {}
        self.src_dsts = defaultdict(set)
        self.node_degree = np.zeros(self.num_nodes, dtype=np.int64)
        self.node_last_time = np.full(self.num_nodes, np.nan, dtype=np.float64)
        self.online_rho = 0.0
        self._rho_initialized = False
        self.random_state = np.random.RandomState(self.seed) if self.seed is not None else np.random.RandomState()

    def _encode_recency(self, last_time: float, current_time: float):
        if not np.isfinite(last_time):
            return -1.0
        return float(np.log1p(max(float(current_time) - float(last_time), 0.0)))

    def _sample_random_dst(self, exclude_dst: int = None):
        if len(self.all_dst_node_ids) == 0:
            raise ValueError("all_dst_node_ids is empty, cannot sample a fallback negative destination!")

        if exclude_dst is None or len(self.all_dst_node_ids) == 1:
            return int(self.random_state.choice(self.all_dst_node_ids))

        for _ in range(8):
            candidate = int(self.random_state.choice(self.all_dst_node_ids))
            if candidate != exclude_dst:
                return candidate

        valid_candidates = self.all_dst_node_ids[self.all_dst_node_ids != exclude_dst]
        if len(valid_candidates) == 0:
            return int(exclude_dst)
        return int(self.random_state.choice(valid_candidates))

    def query(self, src, dst, t):
        src_node_ids = np.asarray(src, dtype=np.int64).reshape(-1)
        dst_node_ids = np.asarray(dst, dtype=np.int64).reshape(-1)
        timestamps = np.asarray(t, dtype=np.float64).reshape(-1)

        if not (len(src_node_ids) == len(dst_node_ids) == len(timestamps)):
            raise ValueError("src, dst and t must have the same batch size!")

        entity_stats = np.zeros((len(src_node_ids), 4), dtype=np.float32)
        pair_stats = np.zeros((len(src_node_ids), 3), dtype=np.float32)

        for idx, (src_node_id, dst_node_id, timestamp) in enumerate(zip(src_node_ids, dst_node_ids, timestamps)):
            entity_stats[idx] = np.array([
                np.log1p(self.node_degree[src_node_id]),
                np.log1p(self.node_degree[dst_node_id]),
                self._encode_recency(self.node_last_time[src_node_id], timestamp),
                self._encode_recency(self.node_last_time[dst_node_id], timestamp),
            ], dtype=np.float32)

            pair_count = self.pair_count.get((int(src_node_id), int(dst_node_id)), 0)
            pair_last_time = self.pair_last_time.get((int(src_node_id), int(dst_node_id)), np.nan)
            pair_stats[idx] = np.array([
                np.log1p(pair_count),
                float(pair_count > 0),
                self._encode_recency(pair_last_time, timestamp),
            ], dtype=np.float32)

        return entity_stats, pair_stats

    def update_batch(self, src, dst, t):
        src_node_ids = np.asarray(src, dtype=np.int64).reshape(-1)
        dst_node_ids = np.asarray(dst, dtype=np.int64).reshape(-1)
        timestamps = np.asarray(t, dtype=np.float64).reshape(-1)

        if not (len(src_node_ids) == len(dst_node_ids) == len(timestamps)):
            raise ValueError("src, dst and t must have the same batch size!")

        if len(src_node_ids) == 0:
            return

        recurrent_flags = np.array(
            [self.pair_count.get((int(src_node_id), int(dst_node_id)), 0) > 0 for src_node_id, dst_node_id in zip(src_node_ids, dst_node_ids)],
            dtype=np.float32,
        )
        batch_recurrence_rate = float(recurrent_flags.mean())
        if not self._rho_initialized:
            self.online_rho = batch_recurrence_rate
            self._rho_initialized = True
        else:
            self.online_rho = self.rho_ema_momentum * self.online_rho + (1.0 - self.rho_ema_momentum) * batch_recurrence_rate

        for src_node_id, dst_node_id, timestamp in zip(src_node_ids, dst_node_ids, timestamps):
            src_node_id = int(src_node_id)
            dst_node_id = int(dst_node_id)
            timestamp = float(timestamp)

            self.pair_count[(src_node_id, dst_node_id)] += 1
            self.pair_last_time[(src_node_id, dst_node_id)] = timestamp
            self.src_dsts[src_node_id].add(dst_node_id)

            self.node_degree[src_node_id] += 1
            self.node_degree[dst_node_id] += 1
            self.node_last_time[src_node_id] = timestamp
            self.node_last_time[dst_node_id] = timestamp

    def sample_hard_neg(self, src, dst, n: int = 1):
        src_node_ids = np.asarray(src, dtype=np.int64)
        dst_node_ids = np.asarray(dst, dtype=np.int64)

        scalar_input = src_node_ids.ndim == 0 and dst_node_ids.ndim == 0
        src_node_ids = src_node_ids.reshape(-1)
        dst_node_ids = dst_node_ids.reshape(-1)

        if len(src_node_ids) != len(dst_node_ids):
            raise ValueError("src and dst must have the same batch size!")
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}!")

        hard_neg_dst_node_ids = np.zeros((len(src_node_ids), n), dtype=np.int64)
        for idx, (src_node_id, dst_node_id) in enumerate(zip(src_node_ids, dst_node_ids)):
            candidates = list(self.src_dsts.get(int(src_node_id), set()) - {int(dst_node_id)})
            if len(candidates) > 0:
                hard_neg_dst_node_ids[idx] = self.random_state.choice(candidates, size=n, replace=len(candidates) < n)
            else:
                hard_neg_dst_node_ids[idx] = np.array(
                    [self._sample_random_dst(exclude_dst=int(dst_node_id)) for _ in range(n)],
                    dtype=np.int64,
                )

        if scalar_input:
            if n == 1:
                return int(hard_neg_dst_node_ids[0, 0])
            return hard_neg_dst_node_ids[0]
        if n == 1:
            return hard_neg_dst_node_ids[:, 0]
        return hard_neg_dst_node_ids

    def state_dict(self):
        return {
            "num_nodes": self.num_nodes,
            "all_dst_node_ids": self.all_dst_node_ids.copy(),
            "rho_ema_momentum": self.rho_ema_momentum,
            "seed": self.seed,
            "pair_count": dict(self.pair_count),
            "pair_last_time": dict(self.pair_last_time),
            "src_dsts": {src_node_id: set(dst_node_ids) for src_node_id, dst_node_ids in self.src_dsts.items()},
            "node_degree": self.node_degree.copy(),
            "node_last_time": self.node_last_time.copy(),
            "online_rho": float(self.online_rho),
            "_rho_initialized": bool(self._rho_initialized),
            "random_state": self.random_state.get_state(),
        }

    def load_state_dict(self, state_dict: dict):
        if int(state_dict["num_nodes"]) != self.num_nodes:
            raise ValueError(f"num_nodes mismatch: expected {self.num_nodes}, got {state_dict['num_nodes']}!")

        self.all_dst_node_ids = np.asarray(state_dict["all_dst_node_ids"], dtype=np.int64).copy()
        self.rho_ema_momentum = float(state_dict["rho_ema_momentum"])
        self.seed = state_dict["seed"]
        self.pair_count = defaultdict(int, state_dict["pair_count"])
        self.pair_last_time = dict(state_dict["pair_last_time"])
        self.src_dsts = defaultdict(set, {src_node_id: set(dst_node_ids) for src_node_id, dst_node_ids in state_dict["src_dsts"].items()})
        self.node_degree = np.asarray(state_dict["node_degree"], dtype=np.int64).copy()
        self.node_last_time = np.asarray(state_dict["node_last_time"], dtype=np.float64).copy()
        self.online_rho = float(state_dict["online_rho"])
        self._rho_initialized = bool(state_dict["_rho_initialized"])
        self.random_state = np.random.RandomState()
        self.random_state.set_state(state_dict["random_state"])

    def clone(self):
        cloned_stats_bank = TemporalStatsBank(
            num_nodes=self.num_nodes,
            all_dst_node_ids=self.all_dst_node_ids,
            rho_ema_momentum=self.rho_ema_momentum,
            seed=self.seed,
        )
        cloned_stats_bank.load_state_dict(self.state_dict())
        return cloned_stats_bank
