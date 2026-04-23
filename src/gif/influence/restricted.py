from __future__ import annotations

import numpy as np
import torch

from .common import hvp
from .projection import as_index_tensor, embed_subset, project_subset


def build_restricted_system(
    model: torch.nn.Module,
    total_loss: torch.Tensor,
    g_full: torch.Tensor,
    index_list: np.ndarray | torch.Tensor,
) -> tuple[torch.Tensor, callable]:
    full_dim = g_full.numel()
    idx = as_index_tensor(index_list, g_full.device)
    rhs = project_subset(hvp(model, total_loss, g_full), idx)

    def a_times(x_sub: torch.Tensor) -> torch.Tensor:
        x_full = embed_subset(x_sub, idx, full_dim)
        return project_subset(
            hvp(model, total_loss, hvp(model, total_loss, x_full)),
            idx,
        )

    return rhs, a_times


def build_restricted_system_from_hvp(
    hvp_fn,
    g_full: torch.Tensor,
    index_list: np.ndarray | torch.Tensor,
) -> tuple[torch.Tensor, callable]:
    full_dim = g_full.numel()
    idx = as_index_tensor(index_list, g_full.device)
    rhs = project_subset(hvp_fn(g_full), idx)

    def a_times(x_sub: torch.Tensor) -> torch.Tensor:
        x_full = embed_subset(x_sub, idx, full_dim)
        return project_subset(hvp_fn(hvp_fn(x_full)), idx)

    return rhs, a_times
