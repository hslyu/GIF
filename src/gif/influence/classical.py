import torch

from gif.influence.common import compute_gradient
from gif.solvers import ihvp


def influence(
    model: torch.nn.Module,
    total_loss: torch.Tensor,
    loss: torch.Tensor,
    mu: float = 1.0,
    tol: float = 1e-6,
    max_iter: int = 200,
    verbose: bool = False,
) -> torch.Tensor:
    gradient = compute_gradient(model, loss)
    return ihvp(
        model=model,
        loss=total_loss,
        v=gradient,
        mu=mu,
        tol=tol,
        max_iter=max_iter,
        verbose=verbose,
    )


class InfluenceFunction:
    def compute(
        self,
        model: torch.nn.Module,
        total_loss: torch.Tensor,
        loss: torch.Tensor,
        mu: float = 1.0,
        tol: float = 1e-6,
        max_iter: int = 200,
        verbose: bool = False,
    ) -> torch.Tensor:
        return influence(
            model=model,
            total_loss=total_loss,
            loss=loss,
            mu=mu,
            tol=tol,
            max_iter=max_iter,
            verbose=verbose,
        )
