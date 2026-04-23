from __future__ import annotations

import numpy as np
import torch

from gif.influence.common import compute_gradient
from gif.influence.restricted import build_restricted_system
from gif.solvers import hyperinf_inverse


def hyperinf_update(
    model: torch.nn.Module,
    total_loss: torch.Tensor,
    target_loss: torch.Tensor,
    index_list: np.ndarray,
    *,
    beta: float | None = None,
    beta_scale: float = 0.9,
    tol: float = 1e-6,
    max_iter: int = 6,
    return_details: bool = False,
    verbose: bool = False,
):
    g_full = compute_gradient(model, target_loss, retain_graph=False)
    rhs, a_times = build_restricted_system(model, total_loss, g_full, index_list)
    result = hyperinf_inverse(
        a_times=a_times,
        rhs=rhs,
        beta=beta,
        beta_scale=beta_scale,
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


class HyperInfluence:
    def compute(
        self,
        model: torch.nn.Module,
        total_loss: torch.Tensor,
        target_loss: torch.Tensor,
        index_list: np.ndarray,
        *,
        beta: float | None = None,
        beta_scale: float = 0.9,
        tol: float = 1e-6,
        max_iter: int = 6,
        return_details: bool = False,
        verbose: bool = False,
    ):
        return hyperinf_update(
            model=model,
            total_loss=total_loss,
            target_loss=target_loss,
            index_list=index_list,
            beta=beta,
            beta_scale=beta_scale,
            tol=tol,
            max_iter=max_iter,
            return_details=return_details,
            verbose=verbose,
        )


HypeInf = HyperInfluence
