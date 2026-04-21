from __future__ import annotations

import numpy as np
import torch

from gif.influence.common import compute_gradient
from gif.influence.restricted import build_restricted_system
from gif.solvers import lissa_inverse


def lissa_update(
    model: torch.nn.Module,
    total_loss: torch.Tensor,
    target_loss: torch.Tensor,
    index_list: np.ndarray,
    *,
    damping: float = 0.0,
    mu: float = 1.0,
    tol: float = 1e-6,
    max_iter: int = 200,
    max_restarts: int = 8,
    return_details: bool = False,
    verbose: bool = False,
):
    g_full = compute_gradient(model, target_loss)
    rhs, a_times = build_restricted_system(model, total_loss, g_full, index_list)
    result = lissa_inverse(
        a_times=a_times,
        rhs=rhs,
        damping=damping,
        mu=mu,
        tol=tol,
        max_iter=max_iter,
        max_restarts=max_restarts,
        return_details=return_details,
        verbose=verbose,
    )
    if return_details:
        return {
            "update": result["solution"],
            "details": result["details"],
        }
    return result


class LiSSAInfluence:
    def compute(
        self,
        model: torch.nn.Module,
        total_loss: torch.Tensor,
        target_loss: torch.Tensor,
        index_list: np.ndarray,
        *,
        damping: float = 0.0,
        mu: float = 1.0,
        tol: float = 1e-6,
        max_iter: int = 200,
        max_restarts: int = 8,
        return_details: bool = False,
        verbose: bool = False,
    ):
        return lissa_update(
            model=model,
            total_loss=total_loss,
            target_loss=target_loss,
            index_list=index_list,
            damping=damping,
            mu=mu,
            tol=tol,
            max_iter=max_iter,
            max_restarts=max_restarts,
            return_details=return_details,
            verbose=verbose,
        )
