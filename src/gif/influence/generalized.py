import numpy as np
import torch

from gif.influence.common import compute_gradient
from gif.solvers import p_lissa


def generalized_influence(
    model: torch.nn.Module,
    total_loss: torch.Tensor,
    target_loss: torch.Tensor,
    index_list: np.ndarray,
    damping: float = 0.0,
    mu: float = 1.0,
    tol: float = 1e-6,
    max_iter: int = 200,
    max_restarts: int = 8,
    verbose: bool = False,
) -> torch.Tensor:
    g_full = compute_gradient(model, target_loss, retain_graph=False)
    return p_lissa(
        model=model,
        loss=total_loss,
        g_full=g_full,
        index_list=index_list,
        damping=damping,
        mu=mu,
        tol=tol,
        max_iter=max_iter,
        max_restarts=max_restarts,
        verbose=verbose,
    )


class GeneralizedInfluence:
    def compute(
        self,
        model: torch.nn.Module,
        total_loss: torch.Tensor,
        target_loss: torch.Tensor,
        index_list: np.ndarray,
        damping: float = 0.0,
        mu: float = 1.0,
        tol: float = 1e-6,
        max_iter: int = 200,
        max_restarts: int = 8,
        verbose: bool = False,
    ) -> torch.Tensor:
        return generalized_influence(
            model=model,
            total_loss=total_loss,
            target_loss=target_loss,
            index_list=index_list,
            damping=damping,
            mu=mu,
            tol=tol,
            max_iter=max_iter,
            max_restarts=max_restarts,
            verbose=verbose,
        )
