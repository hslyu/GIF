from __future__ import annotations

import math
from typing import Callable

import torch

from .iterative import _estimate_lmax_power


TensorOperator = Callable[[torch.Tensor], torch.Tensor]


def _compose_operator(apply_operator: TensorOperator) -> TensorOperator:
    return lambda value: apply_operator(apply_operator(value))


def hyperinf_inverse(
    a_times: TensorOperator,
    rhs: torch.Tensor,
    *,
    beta: float | None = None,
    beta_scale: float = 0.9,
    tol: float = 1e-6,
    max_iter: int = 6,
    return_details: bool = False,
    verbose: bool = False,
):
    if rhs.ndim != 1:
        raise ValueError("rhs must be a flat 1D tensor.")
    if rhs.numel() == 0:
        raise ValueError("rhs must not be empty.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if beta_scale <= 0:
        raise ValueError("beta_scale must be positive.")

    rhs_norm = torch.linalg.norm(rhs).item()
    if rhs_norm == 0.0:
        zero = torch.zeros_like(rhs)
        details = {
            "beta": 0.0,
            "residuals": [0.0],
            "converged": True,
            "iterations": 0,
            "operator_depths": [0],
            "lam_max_hat": 0.0,
        }
        return {"solution": zero, "details": details} if return_details else zero

    device = rhs.device
    dtype = rhs.dtype
    lam_max_hat = _estimate_lmax_power(
        A_times=a_times,
        dim=rhs.numel(),
        device=device,
        dtype=dtype,
    )
    if lam_max_hat is None or not math.isfinite(lam_max_hat) or lam_max_hat <= 0:
        raise RuntimeError("HyperINF failed to estimate a positive spectral scale.")

    resolved_beta = float(beta) if beta is not None else float(beta_scale / lam_max_hat)
    base_residual = lambda value: value - resolved_beta * a_times(value)

    solution = resolved_beta * rhs
    best_solution = solution.detach().clone()
    best_rel_residual = float("inf")
    residual_history: list[float] = []
    operator_depths: list[int] = []
    residual_operator = base_residual
    converged = False

    for iteration in range(max_iter):
        correction = residual_operator(solution)
        candidate = solution + correction

        if not torch.isfinite(candidate).all():
            break

        residual = rhs - a_times(candidate)
        if not torch.isfinite(residual).all():
            break

        rel_residual = torch.linalg.norm(residual).item() / (rhs_norm + 1e-12)
        residual_history.append(rel_residual)
        operator_depths.append(2**iteration)

        if rel_residual < best_rel_residual:
            best_rel_residual = rel_residual
            best_solution = candidate.detach().clone()

        if verbose:
            print(
                f"[hyperinf] iter={iteration + 1}/{max_iter} "
                f"beta={resolved_beta:.3e} rel_residual={rel_residual:.3e}",
                end="\r",
                flush=True,
            )

        solution = candidate
        if rel_residual < tol:
            converged = True
            break

        residual_operator = _compose_operator(residual_operator)

    if verbose:
        print()

    details = {
        "beta": resolved_beta,
        "residuals": residual_history,
        "converged": converged,
        "iterations": len(residual_history),
        "operator_depths": operator_depths,
        "lam_max_hat": lam_max_hat,
    }
    if return_details:
        return {"solution": best_solution, "details": details}
    return best_solution
