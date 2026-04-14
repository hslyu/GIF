import gc

import numpy as np
import torch

from .hessians import compute_gradient, hvp


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
    """
    Compute partial influence function of a given loss

    Parameters:
        model (torch.nn.Module): Model where gradient and hessian is computed.
        total_loss (torch.Tensor): loss where hessian is computed.
        target_loss (torch.Tensor): Loss where gradient is computed.
                                    The original influence function takes data points as input,
                                    but this function takes loss for some generalization issues.
        index_list: list of indices of the parameters where partial influence is computed.
        tol (float): tolerance level for computing inverse hessian-vector product.
        step (float): step size for normalizing the hessian and gradient.

    Returns:
        torch.Tensor: Partial influence function
    """

    normalizer = normalizer
    while True:
        grad = compute_gradient(model, target_loss / normalizer)
        zero_mask = torch.ones(len(grad), dtype=torch.bool, device=grad.device)
        zero_mask[index_list] = False

        grad[zero_mask] = 0
        I_0 = hvp(
            model,
            total_loss / normalizer,
            grad,
        )
        I_0[zero_mask] = 0
        FIF = iphvp_FIF(
            model, total_loss / normalizer, I_0, index_list, tol, max_iter, verbose
        )
        if FIF is not None:
            if verbose:
                print("")
            del I_0
            gc.collect()
            torch.cuda.empty_cache()
            return FIF
        else:
            # if verbose:
            #     print(
            #         f"Normalizer {normalizer:.2f} is too small. Increasing normalizer by {step}."
            #         + " " * 30,
            #         end="\r",
            #         flush=True,
            #     )
            normalizer += step


def iphvp_FIF(
    model: torch.nn.Module,
    loss: torch.Tensor,
    v: torch.Tensor,
    index_list: np.ndarray,
    tol: float = 1e-5,
    max_iter: int = 200,
    verbose: bool = False,
):
    """
    A Simple and Efficient Algorithm for Computing the Pseudo-inverse of partial Hessian-Vector Product.

    Parameters:
        loss (np.ndarray): Empirical risk between true and predicted labels
        model (torch.nn.module): Model where the Hessian and gradient is computed
        v (torch.Tensor): Vector where the Hessian is multiplied
        tol (float): Tolerance level

    Return:
        np.ndarray: Approximation of the IHVP
    """

    def sHVP(
        model: torch.nn.Module,
        loss: torch.Tensor,
        v: torch.Tensor,
    ):
        """
        Subhessian-vector product
        """
        first_HVP = hvp(model, loss, v)
        first_HVP[zero_mask] = 0
        second_HVP = hvp(model, loss, first_HVP)
        second_HVP[zero_mask] = 0

        return second_HVP

    zero_mask = torch.ones(len(v), dtype=torch.bool, device=v.device)
    zero_mask[index_list] = False
    tol = tol * len(index_list) ** 0.5
    # initial settings
    diff = tol + 0.1
    diff_old = 1e10
    I_new = v
    count = 0
    while diff > tol and count < max_iter:
        I_old = I_new
        I_new = v + I_old - sHVP(model, loss, I_old)
        diff = torch.norm(I_new - I_old)
        if count % 2 == 0:
            if diff > diff_old:
                return
            diff_old = diff
        count += 1
        if verbose:
            print(
                f"Computing freeze influence ... [{count}/{max_iter}]",
                end="\r",
                flush=True,
            )

    return I_new[index_list]
