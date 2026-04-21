from __future__ import annotations

import torch


def datainf_inverse_diagonal(
    rhs: torch.Tensor,
    diagonal: torch.Tensor,
    *,
    damping: float = 1e-6,
    return_details: bool = False,
):
    if rhs.ndim != 1 or diagonal.ndim != 1:
        raise ValueError("rhs and diagonal must be flat 1D tensors.")
    if rhs.shape != diagonal.shape:
        raise ValueError("rhs and diagonal must have the same shape.")
    if rhs.numel() == 0:
        raise ValueError("rhs and diagonal must not be empty.")
    if damping < 0:
        raise ValueError("damping must be non-negative.")
    if not torch.isfinite(rhs).all():
        raise ValueError("rhs must be finite.")
    if not torch.isfinite(diagonal).all():
        raise ValueError("diagonal must be finite.")
    if torch.any(diagonal < 0):
        raise ValueError("diagonal must be non-negative.")

    stabilized_diagonal = diagonal + damping
    solution = rhs / stabilized_diagonal
    details = {
        "damping": float(damping),
        "min_diagonal": float(diagonal.min().item()),
        "max_diagonal": float(diagonal.max().item()),
    }
    if return_details:
        return {"solution": solution, "details": details}
    return solution
