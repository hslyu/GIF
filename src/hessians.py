#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Hyeonsu Lyu
# Contact: hslyu4@postech.ac.kr

import gc

import numpy as np
import torch


def compute_hessian(model: torch.nn.Module, loss: torch.Tensor) -> torch.Tensor:
    # # Compute the gradients of the loss w.r.t. the parameters
    # gradients = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    # # Flatten the gradients into a single vector
    # gradients = torch.cat([grad.view(-1) for grad in gradients])
    gradients = compute_gradient(model, loss)
    # Compute the Hessian matrix
    hessian = torch.zeros(gradients.size()[0], gradients.size()[0])
    for idx in range(gradients.size()[0]):
        # Compute the second-order gradients of the loss w.r.t. each parameter
        second_gradients = torch.autograd.grad(
            gradients[idx], list(model.parameters()), retain_graph=True
        )
        # Flatten the second-order gradients into a single vector
        second_gradients = torch.cat(
            [grad.contiguous().view(-1) for grad in second_gradients]
        )
        # Store the second-order gradients in the Hessian matrix
        hessian[idx] = second_gradients

    return hessian


def compute_gradient(model: torch.nn.Module, loss: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [
            grad.view(-1)
            for grad in torch.autograd.grad(
                loss, list(model.parameters()), retain_graph=True
            )
        ]
    )


def ihvp(
    model: torch.nn.Module, loss: torch.Tensor, v: torch.Tensor, tol: float = 1e-4
) -> torch.Tensor:
    """
    A Simple and Efficient Algorithm for Computing the Inverse Hessian-Vector Product.

    References: Koh and Liang. "Understanding Black-box Predictions via Influence Functions."
                Naman Agarwal. "Second-Order Stochastic Optimization for Machine Learning in Linear Time."
                Detail descriptions to be updated on github.

    Parameters:
        loss (np.ndarray): Empirical risk between true and predicted labels
        model (torch.nn.module): model where the Hessian and gradient is computed
        tol (float): Tolerance level

    Return:
        np.ndarray: Approximation of the IHVP
    """

    IHVP_old = v
    IHVP_new = v + IHVP_old - hvp(model, loss, IHVP_old)
    while torch.norm(IHVP_new - IHVP_old) > tol:
        IHVP_old = IHVP_new
        # IHVP_new = v + (I - Hessian) @ IHVP_old
        #          = v + IHVP_old - Hessian @ IHVP_old
        #          = v + IHVP_old - hvp(loss, model, IHVP_old)
        IHVP_new = v + IHVP_old - hvp(model, loss, IHVP_old)

    return IHVP_new


def hvp(
    model: torch.nn.Module,
    loss: torch.Tensor,
    v: torch.Tensor,
    create_graph: bool = True,
) -> torch.Tensor:
    """
    Fast hessian-vector product (HVP) algorithm.
    Reference:  Pearlmutter, B. A. "Fast exact multiplication by the hessian."
                https://stackoverflow.com/questions/74889490/a-faster-hessian-vector-product-in-pytorch

    Parameters:
        loss (torch.Tensor): Evaluated loss from the model
        model (torch.nn.Module): Model where the Hessian is computed

    Returns:
        torch.Tensor: Hessian-vector product
    """

    grads = torch.autograd.grad(
        loss, list(model.parameters()), create_graph=create_graph, retain_graph=True
    )
    grads = torch.cat([grad.view(-1) for grad in grads])
    hvp = torch.autograd.grad(
        grads, list(model.parameters()), grad_outputs=v, retain_graph=True
    )

    return torch.cat([grad.flatten() for grad in hvp])


def influence(
    model: torch.nn.Module, total_loss: torch.Tensor, loss: torch.Tensor
) -> torch.Tensor:
    """
    Compute influence function of a given loss

    Parameters:
        loss (torch.Tensor): loss where gradient is computed.
                             The original influence function takes data points as input,
                             but this function takes loss for some generalization issues.
        total_loss (torch.Tensor): loss where hessian is computed.
        model (torch.nn.Module): Model where gradient and hessian is computed.

    Returns:
        torch.Tensor: Influence function
    """

    return ihvp(model, total_loss, compute_gradient(model, loss))


def generalized_influence(
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
    Compute generalized influence function of a given loss

    Parameters:
        index_list: list of indices of the parameters where generalized influence is computed.
        target_loss (torch.Tensor): Loss where gradient is computed.
                                    The original influence function takes data points as input,
                                    but this function takes loss for some generalization issues.
        total_loss (torch.Tensor): loss where hessian is computed.
        model (torch.nn.Module): Model where gradient and hessian is computed.
        tol (float): tolerance level for computing inverse hessian-vector product.
        step (float): step size for normalizing the hessian and gradient.

    Returns:
        torch.Tensor: generalized influence function
    """

    normalizer = normalizer
    while True:
        v = hvp(
            model,
            total_loss / normalizer,
            compute_gradient(model, target_loss / normalizer),
        )
        PIF = iphvp(
            model,
            total_loss / normalizer,
            v[index_list],
            index_list,
            tol,
            max_iter,
            verbose,
        )
        if PIF is not None:
            if verbose:
                print("")
            del v
            gc.collect()
            torch.cuda.empty_cache()
            return PIF
        else:
            # if verbose: print(
            #         f"Normalizer {normalizer:.2f} is too small. Increasing normalizer by {step}."
            #         + " " * 30,
            #         end="\r",
            #         flush=True,
            #     )
            normalizer += step


def iphvp(
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
        index_list: np.ndarray,
    ):
        """
        Subhessian-vector product
        """
        zeropad_v = torch.zeros(num_params, device=v.device)
        zeropad_v[index_list] = v
        twice_HVP = hvp(model, loss, hvp(model, loss, zeropad_v))

        return twice_HVP[index_list]

    num_params = sum(p.numel() for p in model.parameters())
    tol = tol * len(index_list) ** 0.5
    # initial settings
    diff = tol + 0.1
    diff_old = 1e10
    IHVP_new = v
    count = 0
    while diff > tol and count < max_iter:
        IHVP_old = IHVP_new
        IHVP_new = v + IHVP_old - sHVP(model, loss, IHVP_old, index_list)
        diff = torch.norm(IHVP_new - IHVP_old)
        if count % 2 == 0:
            if diff > diff_old:
                return
            diff_old = diff
        count += 1
        if verbose:
            print(
                f"Computing generalized influence ... [{count}/{max_iter}]",
                end="\r",
                flush=True,
            )

    return IHVP_new
