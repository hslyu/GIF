#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Hyeonsu Lyu
# Contact: hslyu4@postech.ac.kr

import torch


def compute_hessian(loss, model) -> torch.Tensor:
    # # Compute the gradients of the loss w.r.t. the parameters
    # gradients = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    # # Flatten the gradients into a single vector
    # gradients = torch.cat([grad.view(-1) for grad in gradients])
    gradients = compute_gradient(loss, model, True)
    # Compute the Hessian matrix
    hessian = torch.zeros(gradients.size()[0], gradients.size()[0])
    for idx in range(gradients.size()[0]):
        # Compute the second-order gradients of the loss w.r.t. each parameter
        second_gradients = torch.autograd.grad(
            gradients[idx], model.parameters(), retain_graph=True
        )
        # Flatten the second-order gradients into a single vector
        second_gradients = torch.cat(
            [grad.contiguous().view(-1) for grad in second_gradients]
        )
        # Store the second-order gradients in the Hessian matrix
        hessian[idx] = second_gradients

    return hessian


def compute_gradient(loss, model, create_graph=False) -> torch.Tensor:
    return torch.cat(
        [
            grad.view(-1)
            for grad in torch.autograd.grad(
                loss, model.parameters(), create_graph=create_graph, retain_graph=True
            )
        ]
    )


def ihvp(
    loss: torch.Tensor, model: torch.nn.Module, v: torch.Tensor, tol: float = 1e-5
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
    IHVP_new = v + IHVP_old - hvp(loss, model, IHVP_old)
    while torch.norm(IHVP_new - IHVP_old) > tol:
        IHVP_old = IHVP_new
        # IHVP_new = v + (I - Hessian) @ IHVP_old
        #          = v + IHVP_old - Hessian @ IHVP_old
        #          = v + IHVP_old - hvp(loss, model, IHVP_old)
        IHVP_new = v + IHVP_old - hvp(loss, model, IHVP_old)

    return IHVP_new


def hvp(loss: torch.Tensor, model: torch.nn.Module, v: torch.Tensor) -> torch.Tensor:
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

    grads = torch.autograd.grad(loss, list(model.parameters()), create_graph=True)
    grads = torch.cat([grad.view(-1) for grad in grads])
    hvp = torch.autograd.grad(
        grads, list(model.parameters()), grad_outputs=v, retain_graph=True
    )

    return torch.cat([grad.view(-1) for grad in hvp])


def influence(
    loss: torch.Tensor, total_loss: torch.Tensor, model: torch.nn.Module
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

    return ihvp(total_loss, model, compute_gradient(loss, model))


def partial_influence(
    index_list: list,
    loss: torch.Tensor,
    total_loss: torch.Tensor,
    model: torch.nn.Module,
    tol: float = 1e-5,
) -> torch.Tensor:
    """
    Compute partial influence function of a given loss

    Parameters:
        loss (torch.Tensor): loss where gradient is computed.
                             The original influence function takes data points as input,
                             but this function takes loss for some generalization issues.
        total_loss (torch.Tensor): loss where hessian is computed.
        model (torch.nn.Module): Model where gradient and hessian is computed.
        idx (int): index of the parameter where partial influence is computed.

    Returns:
        torch.Tensor: Partial influence function
    """

    v = hvp(total_loss, model, compute_gradient(loss, model))
    PIF = iphvp(index_list, total_loss, model, v[index_list], tol)

    return PIF


def iphvp(
    index_list: list,
    loss: torch.Tensor,
    model: torch.nn.Module,
    v: torch.Tensor,
    tol: float = 1e-5,
) -> torch.Tensor:
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
        num_params = sum(p.numel() for p in model.parameters())
        spanned_v = torch.zeros(num_params, device=v.device)
        spanned_v[index_list] = v
        twice_HVP = hvp(loss, model, hvp(loss, model, spanned_v))

        return twice_HVP[index_list]

    IHVP_old = v
    IHVP_new = v + IHVP_old - sHVP(index_list, loss, model, IHVP_old)
    count = 1
    while torch.norm(IHVP_new - IHVP_old) > tol and count < 10000:
        IHVP_old = IHVP_new
        IHVP_new = v + IHVP_old - sHVP(index_list, loss, model, IHVP_old)
        count += 1

    return IHVP_new
