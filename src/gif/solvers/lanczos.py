from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, eigsh


TensorOperator = Callable[[torch.Tensor], torch.Tensor]


def _materialize_operator(
    a_times: TensorOperator,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    basis = torch.eye(dim, device=device, dtype=dtype)
    return torch.stack([a_times(basis[index]) for index in range(dim)], dim=1)


def lanczos_inverse(
    a_times: TensorOperator,
    rhs: torch.Tensor,
    *,
    rank: int | None = None,
    damping: float = 0.0,
    tol: float = 1e-6,
    max_iter: int = 20,
    return_details: bool = False,
):
    if rhs.ndim != 1:
        raise ValueError("rhs must be a flat 1D tensor.")
    if rhs.numel() == 0:
        raise ValueError("rhs must not be empty.")
    if damping < 0:
        raise ValueError("damping must be non-negative.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")

    dim = rhs.numel()
    if rank is None:
        rank = min(max(1, dim // 4), dim)
    if rank <= 0:
        raise ValueError("rank must be positive.")

    rhs_norm = torch.linalg.norm(rhs).item()
    if rhs_norm == 0.0:
        zero = torch.zeros_like(rhs)
        details = {
            "rank": 0,
            "damping": float(damping),
            "residual_norm": 0.0,
            "relative_residual": 0.0,
            "eigenvalues": [],
            "used_dense_fallback": False,
        }
        return {"solution": zero, "details": details} if return_details else zero

    device = rhs.device
    dtype = rhs.dtype

    used_dense_fallback = rank >= dim or dim == 1
    if used_dense_fallback:
        operator = _materialize_operator(a_times, dim=dim, device=device, dtype=dtype)
        stabilized = operator + damping * torch.eye(dim, device=device, dtype=dtype)
        solution = torch.linalg.solve(stabilized, rhs)
        eigenvalues = torch.linalg.eigvalsh(operator).detach().cpu().tolist()
    else:
        scipy_dtype = np.float64 if dtype == torch.float64 else np.float32

        def scipy_apply(vec: np.ndarray) -> np.ndarray:
            value = torch.from_numpy(vec).to(device=device, dtype=dtype)
            output = a_times(value)
            return output.detach().cpu().numpy().astype(scipy_dtype, copy=False)

        operator = LinearOperator(
            shape=(dim, dim),
            matvec=scipy_apply,
            dtype=scipy_dtype,
        )
        k = min(rank, dim - 1)
        ncv = min(dim, max(2 * k + 1, 20))
        eigenvalues, eigenvectors = eigsh(
            A=operator,
            k=k,
            which="LA",
            maxiter=max_iter,
            tol=tol,
            ncv=ncv,
        )
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        eigenvalues_tensor = torch.from_numpy(eigenvalues).to(device=device, dtype=dtype)
        eigenvectors_tensor = torch.from_numpy(eigenvectors).to(device=device, dtype=dtype)
        coeffs = eigenvectors_tensor.T @ rhs
        stabilized = eigenvalues_tensor + damping
        solution = eigenvectors_tensor @ (coeffs / stabilized)
        eigenvalues = eigenvalues.tolist()

    residual = rhs - a_times(solution)
    if damping > 0:
        residual = residual - damping * solution
    residual_norm = torch.linalg.norm(residual).item()
    details = {
        "rank": int(min(rank, dim)),
        "damping": float(damping),
        "residual_norm": residual_norm,
        "relative_residual": residual_norm / (rhs_norm + 1e-12),
        "eigenvalues": [float(value) for value in eigenvalues],
        "used_dense_fallback": used_dense_fallback,
    }
    if return_details:
        return {"solution": solution, "details": details}
    return solution
