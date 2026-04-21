from __future__ import annotations

import numpy as np
import torch

from gif.influence.common import compute_gradient
from gif.influence.restricted import build_restricted_system
from gif.solvers import lanczos_inverse


def lanczos_update(
    model: torch.nn.Module,
    total_loss: torch.Tensor,
    target_loss: torch.Tensor,
    index_list: np.ndarray,
    *,
    rank: int | None = None,
    damping: float = 0.0,
    tol: float = 1e-6,
    max_iter: int = 20,
    return_details: bool = False,
):
    g_full = compute_gradient(model, target_loss)
    rhs, a_times = build_restricted_system(model, total_loss, g_full, index_list)
    result = lanczos_inverse(
        a_times=a_times,
        rhs=rhs,
        rank=rank,
        damping=damping,
        tol=tol,
        max_iter=max_iter,
        return_details=return_details,
    )
    if return_details:
        return {
            "update": result["solution"],
            "details": result["details"],
        }
    return result


class LanczosInfluence:
    def compute(
        self,
        model: torch.nn.Module,
        total_loss: torch.Tensor,
        target_loss: torch.Tensor,
        index_list: np.ndarray,
        *,
        rank: int | None = None,
        damping: float = 0.0,
        tol: float = 1e-6,
        max_iter: int = 20,
        return_details: bool = False,
    ):
        return lanczos_update(
            model=model,
            total_loss=total_loss,
            target_loss=target_loss,
            index_list=index_list,
            rank=rank,
            damping=damping,
            tol=tol,
            max_iter=max_iter,
            return_details=return_details,
        )
