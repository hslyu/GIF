#!/usr/bin/env python3

# Author: Hyeonsu Lyu
# Contact: hslyu4@postech.ac.kr
# Implementation of the paper "Deeper Understanding of Black-box Predictions via Generalized Influence Functions"

import gc

import torch

from .hessians import compute_gradient, hvp, ihvp


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
    """
    Compute second-order influence function based on the paper
    "in the On Second-Order Group Influence Functions for Black-Box Predictions"

    Parameters:
        model (torch.nn.Module): Model where gradient and hessian is computed.
        total_loss (torch.Tensor): loss where hessian is computed.
        target_loss (torch.Tensor): Loss where gradient is computed.
                                    The original influence function takes data points as input,
                                    but this function takes loss for some generalization issues.
        num_total_data (int): |S|, number of total data points.
        num_target_data (int): |U|, number of target data points.
        tol (float): tolerance level for computing inverse hessian-vector product.
        step (float): step size for normalizing the hessian and gradient.
        max_iter (int): maximum number of iterations for computing inverse hessian-vector product.
        verbose (bool): whether to print the progress.
        normalizer (float): initial normalizer for the hessian-vector product.

    Returns:
        torch.Tensor: influence function dervied from the second-order approximation.
    """
    ratio = num_target_data / num_total_data

    # Eq. (12)
    # target_loss is average loss, so we need to multiply by the number of target data points |U|
    # Then, |U| / |S| = ratio
    first_order_influence = (
        plain_influence(model, total_loss, target_loss, tol, step, max_iter, verbose)
        * ratio
        / (1 - ratio)
    )

    # IHVP series is implemented by referring to Eq (12) of "Deeper Understanding of Black-box Predictions via Generalized Influence Functions"
    normalizer = normalizer
    while True:
        # Terms in the parenthesis of the RHS in Eq. (15)
        I_0 = hvp(model, total_loss - target_loss, first_order_influence)
        # inverse of LHS in Eq. (15)
        second_order_influence = ihvp(
            model,
            total_loss / normalizer,
            I_0 / normalizer,
            tol,
            max_iter,
        )
        if second_order_influence is not None:
            second_order_influence *= ratio / (
                1 - ratio
            )  # Eq. (15) |U| / (|S| - |U|) = (|U|/|S|) / (1 - |U|/|S|)
            if verbose:
                print("")
            del I_0
            gc.collect()
            torch.cuda.empty_cache()
            return first_order_influence + second_order_influence
        else:
            # if verbose:
            #     print(
            #         f"Normalizer {normalizer:.2f} is too small. Increasing normalizer by {step}."
            #         + " " * 30,
            #         end="\r",
            #         flush=True,
            #     )
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
    """
    Compute influence function of a given loss.
    Implementation of the influence function in the paper "Understanding Black-box Predictions via Influence Functions"

    Parameters:
        model (torch.nn.Module): Model where gradient and hessian is computed.
        total_loss (torch.Tensor): loss where hessian is computed.
        target_loss (torch.Tensor): loss where gradient is computed.
                             The original influence function takes data points as input,
                             but this function takes loss for some generalization issues.
        tol (float): tolerance level for computing inverse hessian-vector product.
        step (float): step size for normalizing the hessian and gradient.
        max_iter (int): maximum number of iterations for computing inverse hessian-vector product.
        verbose (bool): whether to print the progress.
        normalizer (float): initial normalizer for the hessian-vector product.

    Returns:
        torch.Tensor: Influence function
    """
    normalizer = normalizer
    while True:
        I_0 = hvp(
            model,
            total_loss / normalizer,
            compute_gradient(model, target_loss / normalizer),
        )
        GIF = ihvp(
            model,
            total_loss / normalizer,
            I_0,
            tol,
            max_iter,
            verbose,
        )
        if GIF is not None:
            if verbose:
                print("")
            del I_0
            gc.collect()
            torch.cuda.empty_cache()
            return GIF
        else:
            # if verbose:
            #     print(
            #         f"Normalizer {normalizer:.2f} is too small. Increasing normalizer by {step}."
            #         + " " * 30,
            #         end="\r",
            #         flush=True,
            #     )
            normalizer += step
