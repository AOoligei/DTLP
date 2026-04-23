import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor):
        return self.layers(x)


class DualPathLinkPredictor(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int = None):
        super().__init__()

        if input_dim < 2:
            raise ValueError(f"input_dim must be at least 2, got {input_dim}!")

        self.input_dim = input_dim
        self.projected_dim = input_dim // 2
        self.hidden_dim = hidden_dim if hidden_dim is not None else input_dim

        self.entity_projector = nn.Linear(input_dim, self.projected_dim)
        self.pair_projector = nn.Linear(input_dim, self.projected_dim)

        self.entity_head = MLP(input_dim=2 * self.projected_dim + 4, hidden_dim=self.hidden_dim, output_dim=1)
        self.pair_head = MLP(input_dim=2 * self.projected_dim + 3, hidden_dim=self.hidden_dim, output_dim=1)

        self._cached_projection_pairs = []

    def reset_cached_projections(self):
        self._cached_projection_pairs = []

    def orthogonality_loss(self):
        if len(self._cached_projection_pairs) == 0:
            return next(self.parameters()).new_zeros(())

        losses = []
        for entity_projection, pair_projection in self._cached_projection_pairs:
            normalized_entity_projection = F.normalize(entity_projection, p=2, dim=-1, eps=1e-8)
            normalized_pair_projection = F.normalize(pair_projection, p=2, dim=-1, eps=1e-8)
            losses.append((normalized_entity_projection * normalized_pair_projection).sum(dim=-1).pow(2).mean())

        self._cached_projection_pairs = []
        return torch.stack(losses).mean()

    def forward(self, src_emb: torch.Tensor, dst_emb: torch.Tensor, entity_stats: torch.Tensor, pair_stats: torch.Tensor):
        if entity_stats.dim() == 1:
            entity_stats = entity_stats.unsqueeze(dim=0)
        if pair_stats.dim() == 1:
            pair_stats = pair_stats.unsqueeze(dim=0)

        entity_stats = entity_stats.to(device=src_emb.device, dtype=src_emb.dtype)
        pair_stats = pair_stats.to(device=src_emb.device, dtype=src_emb.dtype)

        src_entity_projection = self.entity_projector(src_emb)
        dst_entity_projection = self.entity_projector(dst_emb)
        src_pair_projection = self.pair_projector(src_emb)
        dst_pair_projection = self.pair_projector(dst_emb)

        if self.training:
            self._cached_projection_pairs.extend([
                (src_entity_projection, src_pair_projection),
                (dst_entity_projection, dst_pair_projection),
            ])

        entity_input = torch.cat([src_entity_projection, dst_entity_projection, entity_stats], dim=1)
        pair_input = torch.cat([src_pair_projection, dst_pair_projection, pair_stats], dim=1)

        s_entity = self.entity_head(entity_input).squeeze(dim=-1)
        s_pair = self.pair_head(pair_input).squeeze(dim=-1)
        s_total = s_entity + s_pair

        return s_entity, s_pair, s_total
