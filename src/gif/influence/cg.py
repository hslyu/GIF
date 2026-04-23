from __future__ import annotations

import numpy as np
import torch

from gif.influence.common import compute_gradient
from gif.influence.restricted import build_restricted_system
from gif.solvers import cg_inverse


def cg_update(
    model: torch.nn.Module,
    total_loss: torch.Tensor,
    target_loss: torch.Tensor,
    index_list: np.ndarray,
    *,
    damping: float = 0.0,
    tol: float = 1e-6,
    max_iter: int = 200,
    return_details: bool = False,
    verbose: bool = False,
):
    g_full = compute_gradient(model, target_loss, retain_graph=False)
    rhs, a_times = build_restricted_system(model, total_loss, g_full, index_list)
    result = cg_inverse(
        a_times=a_times,
        rhs=rhs,
        damping=damping,
        tol=tol,
        max_iter=max_iter,
        return_details=return_details,
        verbose=verbose,
    )
    if return_details:
        return {
            "update": result["solution"],
            "details": result["details"],
        }
    return result


class CGInfluence:
    def compute(
        self,
        model: torch.nn.Module,
        total_loss: torch.Tensor,
        target_loss: torch.Tensor,
        index_list: np.ndarray,
        *,
        damping: float = 0.0,
        tol: float = 1e-6,
        max_iter: int = 200,
        return_details: bool = False,
        verbose: bool = False,
    ):
        return cg_update(
            model=model,
            total_loss=total_loss,
            target_loss=target_loss,
            index_list=index_list,
            damping=damping,
            tol=tol,
            max_iter=max_iter,
            return_details=return_details,
            verbose=verbose,
        )
