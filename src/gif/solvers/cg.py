from __future__ import annotations

from typing import Callable

import torch


TensorOperator = Callable[[torch.Tensor], torch.Tensor]


def cg_inverse(
    a_times: TensorOperator,
    rhs: torch.Tensor,
    *,
    damping: float = 0.0,
    tol: float = 1e-6,
    max_iter: int = 200,
    return_details: bool = False,
    verbose: bool = False,
):
    if rhs.ndim != 1:
        raise ValueError("rhs must be a flat 1D tensor.")
    if rhs.numel() == 0:
        raise ValueError("rhs must not be empty.")
    if damping < 0:
        raise ValueError("damping must be non-negative.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")

    rhs_norm = torch.linalg.norm(rhs).item()
    if rhs_norm == 0.0:
        zero = torch.zeros_like(rhs)
        details = {
            "damping": float(damping),
            "residuals": [0.0],
            "converged": True,
            "iterations": 0,
        }
        return {"solution": zero, "details": details} if return_details else zero

    def apply_operator(value: torch.Tensor) -> torch.Tensor:
        out = a_times(value)
        if damping > 0:
            out = out + damping * value
        return out

    solution = torch.zeros_like(rhs)
    residual = rhs.clone()
    direction = residual.clone()
    residual_dot = torch.dot(residual, residual)
    residual_history = [torch.sqrt(residual_dot).item() / (rhs_norm + 1e-12)]
    converged = residual_history[-1] < tol

    if converged:
        details = {
            "damping": float(damping),
            "residuals": residual_history,
            "converged": True,
            "iterations": 0,
        }
        if return_details:
            return {"solution": solution, "details": details}
        return solution

    for iteration in range(max_iter):
        Ad = apply_operator(direction)
        denom = torch.dot(direction, Ad)
        if not torch.isfinite(denom) or denom.item() <= 0:
            raise RuntimeError("cg_inverse failed: operator is not positive definite.")

        alpha = residual_dot / denom
        candidate = solution + alpha * direction
        if not torch.isfinite(candidate).all():
            raise RuntimeError("cg_inverse failed: non-finite iterate.")

        residual_next = residual - alpha * Ad
        rel_residual = torch.linalg.norm(residual_next).item() / (rhs_norm + 1e-12)
        residual_history.append(rel_residual)

        if verbose:
            print(
                f"[cg] iter={iteration + 1}/{max_iter} rel_residual={rel_residual:.3e}",
                end="\r",
                flush=True,
            )

        solution = candidate
        if rel_residual < tol:
            converged = True
            break

        residual_dot_next = torch.dot(residual_next, residual_next)
        beta = residual_dot_next / residual_dot
        direction = residual_next + beta * direction
        residual = residual_next
        residual_dot = residual_dot_next

    if verbose:
        print()

    details = {
        "damping": float(damping),
        "residuals": residual_history,
        "converged": converged,
        "iterations": len(residual_history) - 1,
    }
    if return_details:
        return {"solution": solution, "details": details}
    return solution
