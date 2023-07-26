import time

import numpy as np
import torch

from .hessians import compute_gradient, hvp


def freeze_influence(
    index_list: np.ndarray,
    target_loss: torch.Tensor,
    total_loss: torch.Tensor,
    model: torch.nn.Module,
    tol: float = 1e-4,
    step: float = 0.5,
    verbose: bool = False,
    normalizer: float = 1,
) -> torch.Tensor:
    """
    Compute partial influence function of a given loss

    Parameters:
        index_list: list of indices of the parameters where partial influence is computed.
        target_loss (torch.Tensor): Loss where gradient is computed.
                                    The original influence function takes data points as input,
                                    but this function takes loss for some generalization issues.
        total_loss (torch.Tensor): loss where hessian is computed.
        model (torch.nn.Module): Model where gradient and hessian is computed.
        tol (float): tolerance level for computing inverse hessian-vector product.
        step (float): step size for normalizing the hessian and gradient.

    Returns:
        torch.Tensor: Partial influence function
    """

    num_params = sum(p.numel() for p in model.parameters())
    normalizer = normalizer
    while True:
        zeropad_grad = _zeropadding(
            compute_gradient(target_loss / normalizer, model), index_list, num_params
        )
        v = hvp(
            total_loss / normalizer,
            model,
            zeropad_grad,
        )
        v = _zeropadding(v, index_list, num_params)
        PIF = iphvp(index_list, total_loss / normalizer, model, v, tol)
        if PIF is not None:
            print("")
            return PIF
        else:
            if verbose:
                print(
                    f"Normalizer {normalizer:.2f} is too small. Increasing normalizer by {step}."
                    + " " * 30,
                    end="\r",
                    flush=True,
                )
            normalizer += step


def iphvp(
    index_list: np.ndarray,
    loss: torch.Tensor,
    model: torch.nn.Module,
    v: torch.Tensor,
    tol: float = 1e-5,
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

    def sHVP(index_list, loss, model, v):
        """
        Subhessian-vector product
        """
        first_HVP = _zeropadding(hvp(loss, model, v, True), index_list, num_params)
        second_HVP = hvp(loss, model, first_HVP, True)

        return _zeropadding(second_HVP, index_list, num_params)

    num_params = sum(p.numel() for p in model.parameters())
    tol = tol * len(index_list) ** 0.5
    # initial settings
    diff = tol + 0.1
    diff_old = 1e10
    IHVP_new = v
    count = 0
    elapsed_time = 0
    while diff > tol and count < 10000:
        start = time.time()
        IHVP_old = IHVP_new
        IHVP_new = v + IHVP_old - sHVP(index_list, loss, model, IHVP_old)
        diff = torch.norm(IHVP_new - IHVP_old)
        if count % 2 == 0:
            if diff > diff_old:
                return
            diff_old = diff
        elapsed_time += time.time() - start
        count += 1
        print(
            f"Computing partial influence ... [{count}/10000], Tolerance: {diff/ len(index_list) ** 0.5:.3E}, Avg. computing time: {elapsed_time/count:.3f}s"
            + " " * 10,
            end="\r",
            flush=True,
        )

    return IHVP_new[index_list]


def _zeropadding(v: torch.Tensor, index_list, num_params) -> torch.Tensor:
    """
    Padding zeros but for params in index_list
    """
    zeropad_v = torch.zeros(num_params, device=v.device)
    zeropad_v[index_list] = v[index_list]
    return zeropad_v
