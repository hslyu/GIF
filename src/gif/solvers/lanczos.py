from __future__ import annotations

from typing import Callable

import torch


TensorOperator = Callable[[torch.Tensor], torch.Tensor]


def _materialize_operator(
    a_times: TensorOperator,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    basis = torch.eye(dim, device=device, dtype=dtype)
    return torch.stack([a_times(basis[index]) for index in range(dim)], dim=1)


def _lanczos_inverse_approximation(
    a_times: TensorOperator,
    rhs: torch.Tensor,
    *,
    rank: int,
    damping: float,
    tol: float,
    max_iter: int,
) -> tuple[torch.Tensor, list[float], int]:
    dim = rhs.numel()
    rhs_norm = torch.linalg.norm(rhs)
    max_steps = min(rank, max_iter, dim)

    def apply_operator(value: torch.Tensor) -> torch.Tensor:
        output = a_times(value)
        if damping > 0:
            output = output + damping * value
        return output

    q = rhs / rhs_norm
    q_prev = torch.zeros_like(rhs)
    beta_prev = torch.zeros((), device=rhs.device, dtype=rhs.dtype)
    q_vectors: list[torch.Tensor] = []
    alphas: list[torch.Tensor] = []
    betas: list[torch.Tensor] = []

    for step in range(max_steps):
        z = apply_operator(q)
        if step > 0:
            z = z - beta_prev * q_prev

        alpha = torch.dot(q, z)
        z = z - alpha * q

        # Full reorthogonalization keeps the GPU Lanczos basis stable for
        # small-to-medium ranks used in these benchmarks.
        for q_old in q_vectors:
            z = z - torch.dot(q_old, z) * q_old

        beta = torch.linalg.norm(z)
        q_vectors.append(q)
        alphas.append(alpha)

        if beta <= tol * max(rhs_norm.item(), 1.0):
            break

        betas.append(beta)
        q_prev = q
        q = z / beta
        beta_prev = beta

    basis_size = len(q_vectors)
    Q = torch.stack(q_vectors, dim=1)
    T = torch.zeros(
        basis_size,
        basis_size,
        device=rhs.device,
        dtype=rhs.dtype,
    )
    diag = torch.stack(alphas)
    T.diagonal().copy_(diag)
    if basis_size > 1:
        off_diag = torch.stack(betas[: basis_size - 1])
        idx = torch.arange(basis_size - 1, device=rhs.device)
        T[idx, idx + 1] = off_diag
        T[idx + 1, idx] = off_diag

    e1 = torch.zeros(basis_size, device=rhs.device, dtype=rhs.dtype)
    e1[0] = rhs_norm
    eigenvalues_tensor, eigenvectors_tensor = torch.linalg.eigh(T)
    coeffs = eigenvectors_tensor.T @ e1
    abs_eigenvalues = torch.abs(eigenvalues_tensor)
    eigen_floor = max(float(tol), torch.finfo(rhs.dtype).eps) * max(
        float(abs_eigenvalues.max().item()),
        1.0,
    )
    keep = abs_eigenvalues > eigen_floor
    inverse_coeffs = torch.zeros_like(coeffs)
    inverse_coeffs[keep] = coeffs[keep] / eigenvalues_tensor[keep]
    projected_solution = eigenvectors_tensor @ inverse_coeffs

    eigenvalues = eigenvalues_tensor.detach().cpu().tolist()
    solution = Q @ projected_solution
    return solution, [float(value) for value in eigenvalues], basis_size


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
        solution, eigenvalues, rank = _lanczos_inverse_approximation(
            a_times=a_times,
            rhs=rhs,
            rank=min(rank, dim),
            damping=damping,
            tol=tol,
            max_iter=max_iter,
        )

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
