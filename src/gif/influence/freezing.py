import gc

import numpy as np
import torch

from gif.influence.common import compute_gradient, hvp


def freeze_influence(
    model: torch.nn.Module,
    total_loss: torch.Tensor,
    target_loss: torch.Tensor,
    index_list: np.ndarray,
    tol: float = 1e-4,
    step: float = 0.5,
    max_iter: int = 200,
    verbose: bool = False,
    normalizer: float = 1,
) -> torch.Tensor:
    while True:
        grad = compute_gradient(model, target_loss / normalizer)
        zero_mask = torch.ones(len(grad), dtype=torch.bool, device=grad.device)
        zero_mask[index_list] = False

        grad[zero_mask] = 0
        initial = hvp(
            model,
            total_loss / normalizer,
            grad,
        )
        initial[zero_mask] = 0
        frozen = iphvp_fif(
            model, total_loss / normalizer, initial, index_list, tol, max_iter, verbose
        )
        if frozen is not None:
            if verbose:
                print("")
            del initial
            gc.collect()
            torch.cuda.empty_cache()
            return frozen
        normalizer += step


def iphvp_fif(
    model: torch.nn.Module,
    loss: torch.Tensor,
    v: torch.Tensor,
    index_list: np.ndarray,
    tol: float = 1e-5,
    max_iter: int = 200,
    verbose: bool = False,
):
    def sub_hvp(
        current_model: torch.nn.Module,
        current_loss: torch.Tensor,
        current_vector: torch.Tensor,
    ):
        first_hvp = hvp(current_model, current_loss, current_vector)
        first_hvp[zero_mask] = 0
        second_hvp = hvp(current_model, current_loss, first_hvp)
        second_hvp[zero_mask] = 0
        return second_hvp

    zero_mask = torch.ones(len(v), dtype=torch.bool, device=v.device)
    zero_mask[index_list] = False
    tol = tol * len(index_list) ** 0.5
    diff = tol + 0.1
    diff_old = 1e10
    current = v
    count = 0
    while diff > tol and count < max_iter:
        previous = current
        current = v + previous - sub_hvp(model, loss, previous)
        diff = torch.norm(current - previous)
        if count % 2 == 0:
            if diff > diff_old:
                return None
            diff_old = diff
        count += 1
        if verbose:
            print(
                f"Computing freeze influence ... [{count}/{max_iter}]",
                end="\r",
                flush=True,
            )

    return current[index_list]


class FreezingInfluence:
    def compute(
        self,
        model: torch.nn.Module,
        total_loss: torch.Tensor,
        target_loss: torch.Tensor,
        index_list: np.ndarray,
        tol: float = 1e-4,
        step: float = 0.5,
        max_iter: int = 200,
        verbose: bool = False,
        normalizer: float = 1,
    ) -> torch.Tensor:
        return freeze_influence(
            model=model,
            total_loss=total_loss,
            target_loss=target_loss,
            index_list=index_list,
            tol=tol,
            step=step,
            max_iter=max_iter,
            verbose=verbose,
            normalizer=normalizer,
        )
