import gc

import torch

from gif.influence.classical import influence
from gif.influence.common import compute_gradient, hvp
from gif.solvers import ihvp


def second_influence(
    model: torch.nn.Module,
    total_loss: torch.Tensor,
    target_loss: torch.Tensor,
    num_total_data: int,
    num_target_data: int,
    tol: float = 1e-4,
    step: float = 0.5,
    max_iter: int = 200,
    verbose: bool = False,
    normalizer: float = 1,
) -> torch.Tensor:
    ratio = num_target_data / num_total_data
    first_order = (
        plain_influence(model, total_loss, target_loss, tol, step, max_iter, verbose)
        * ratio
        / (1 - ratio)
    )

    while True:
        initial = hvp(model, total_loss - target_loss, first_order)
        second_order = ihvp(
            model,
            total_loss / normalizer,
            initial / normalizer,
            tol=tol,
            max_iter=max_iter,
        )
        if second_order is not None:
            second_order *= ratio / (1 - ratio)
            if verbose:
                print("")
            del initial
            gc.collect()
            torch.cuda.empty_cache()
            return first_order + second_order
        normalizer += step


def plain_influence(
    model: torch.nn.Module,
    total_loss: torch.Tensor,
    target_loss: torch.Tensor,
    tol: float = 1e-4,
    step: float = 0.5,
    max_iter: int = 200,
    verbose: bool = False,
    normalizer: float = 1,
) -> torch.Tensor:
    while True:
        initial = hvp(
            model,
            total_loss / normalizer,
            compute_gradient(model, target_loss / normalizer),
        )
        approx = ihvp(
            model,
            total_loss / normalizer,
            initial,
            tol=tol,
            max_iter=max_iter,
            verbose=verbose,
        )
        if approx is not None:
            if verbose:
                print("")
            del initial
            gc.collect()
            torch.cuda.empty_cache()
            return approx
        normalizer += step


class SecondOrderInfluence:
    def compute(
        self,
        model: torch.nn.Module,
        total_loss: torch.Tensor,
        target_loss: torch.Tensor,
        num_total_data: int,
        num_target_data: int,
        tol: float = 1e-4,
        step: float = 0.5,
        max_iter: int = 200,
        verbose: bool = False,
        normalizer: float = 1,
    ) -> torch.Tensor:
        return second_influence(
            model=model,
            total_loss=total_loss,
            target_loss=target_loss,
            num_total_data=num_total_data,
            num_target_data=num_target_data,
            tol=tol,
            step=step,
            max_iter=max_iter,
            verbose=verbose,
            normalizer=normalizer,
        )
