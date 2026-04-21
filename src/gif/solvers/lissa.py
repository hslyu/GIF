from __future__ import annotations

from typing import Callable

import torch

from .iterative import _estimate_lmax_power


TensorOperator = Callable[[torch.Tensor], torch.Tensor]


def lissa_inverse(
    a_times: TensorOperator,
    rhs: torch.Tensor,
    *,
    damping: float = 0.0,
    mu: float = 1.0,
    tol: float = 1e-6,
    max_iter: int = 200,
    max_restarts: int = 8,
    return_details: bool = False,
    verbose: bool = False,
):
    if rhs.ndim != 1:
        raise ValueError("rhs must be a flat 1D tensor.")
    if rhs.numel() == 0:
        raise ValueError("rhs must not be empty.")
    if damping < 0:
        raise ValueError("damping must be non-negative.")
    if mu <= 0:
        raise ValueError("mu must be positive.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if max_restarts <= 0:
        raise ValueError("max_restarts must be positive.")

    rhs_norm = torch.linalg.norm(rhs).item()
    if rhs_norm == 0.0:
        zero = torch.zeros_like(rhs)
        details = {
            "mu": 0.0,
            "damping": float(damping),
            "residuals": [0.0],
            "converged": True,
            "iterations": 0,
            "restarts": 0,
            "lam_max_hat": 0.0,
        }
        return {"solution": zero, "details": details} if return_details else zero

    eps = 1e-12
    stable_patience = 2
    bad_patience = 2
    stall_window = 8
    min_progress = 1e-3
    blowup_factor = 5.0
    x_blowup = 1e4

    def apply_operator(value: torch.Tensor) -> torch.Tensor:
        out = a_times(value)
        if damping > 0:
            out = out + damping * value
        return out

    lam_max_hat = _estimate_lmax_power(
        A_times=apply_operator,
        dim=rhs.numel(),
        device=rhs.device,
        dtype=rhs.dtype,
    )
    if lam_max_hat is not None:
        mu = min(mu, 0.9 / max(lam_max_hat, eps))

    best_x = None
    best_rel_residual = float("inf")
    residual_history: list[float] = []
    total_iterations = 0
    converged = False
    restarts_used = 0

    for restart in range(max_restarts):
        x = mu * rhs.clone()
        x0_norm = torch.linalg.norm(x).item()
        ema_residual = None
        best_ema = float("inf")
        stable_hits = 0
        bad_hits = 0
        recent_ema: list[float] = []

        for iteration in range(max_iter):
            Ax = apply_operator(x)
            residual = rhs - Ax
            step = mu * residual
            x_next = x + step
            total_iterations += 1

            if (
                (not torch.isfinite(residual).all())
                or (not torch.isfinite(x).all())
                or (not torch.isfinite(x_next).all())
            ):
                bad_hits += 1
                x = x_next
                continue

            residual_norm = torch.linalg.norm(residual).item()
            rel_residual = residual_norm / (rhs_norm + eps)
            residual_history.append(rel_residual)
            step_norm = torch.linalg.norm(step).item()
            x_norm = torch.linalg.norm(x).item()

            if ema_residual is None:
                ema_residual = residual_norm
            else:
                ema_residual = 0.9 * ema_residual + 0.1 * residual_norm

            best_ema = min(best_ema, ema_residual)

            if rel_residual < best_rel_residual:
                best_rel_residual = rel_residual
                best_x = x_next.detach().clone()

            residual_ok = rel_residual < tol
            step_ok = step_norm <= tol * max(x_norm, 1.0)
            if residual_ok and step_ok:
                stable_hits += 1
            else:
                stable_hits = 0

            if stable_hits >= stable_patience:
                converged = True
                x = x_next
                break

            if ema_residual > blowup_factor * max(best_ema, 1e-30):
                bad_hits += 1
            else:
                bad_hits = 0

            if x_norm > x_blowup * max(x0_norm, rhs_norm, 1.0):
                bad_hits += 1

            if bad_hits >= bad_patience:
                mu *= 0.5
                restarts_used = restart + 1
                if verbose:
                    print(
                        f"[lissa] divergence at restart={restart}, "
                        f"iter={iteration + 1}, new_mu={mu:.3e}"
                    )
                break

            recent_ema.append(ema_residual)
            if len(recent_ema) > stall_window:
                recent_ema.pop(0)

            if len(recent_ema) == stall_window:
                progress = (recent_ema[0] - recent_ema[-1]) / max(recent_ema[0], 1e-30)
                if progress < min_progress:
                    mu *= 0.5
                    restarts_used = restart + 1
                    if verbose:
                        print(
                            f"[lissa] stagnation at restart={restart}, "
                            f"iter={iteration + 1}, new_mu={mu:.3e}"
                        )
                    break

            if verbose:
                print(
                    f"[lissa] restart={restart} iter={iteration + 1}/{max_iter} "
                    f"mu={mu:.3e} rel_residual={rel_residual:.3e}",
                    end="\r",
                    flush=True,
                )

            x = x_next
        else:
            if best_x is None:
                best_x = x.detach().clone()
            break

        if converged:
            break

    if verbose:
        print()

    if best_x is None:
        raise RuntimeError("lissa_inverse failed: no finite iterate was produced.")

    details = {
        "mu": float(mu),
        "damping": float(damping),
        "residuals": residual_history,
        "converged": converged,
        "iterations": total_iterations,
        "restarts": restarts_used,
        "lam_max_hat": 0.0 if lam_max_hat is None else float(lam_max_hat),
    }
    if return_details:
        return {"solution": best_x, "details": details}
    return best_x
