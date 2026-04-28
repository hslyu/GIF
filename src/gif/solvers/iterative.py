import numpy as np
import torch

from gif.influence.common import hvp
from gif.influence.projection import _as_index_tensor, _embed_subset, _project_subset


def _estimate_lmax_power(
    A_times,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
    num_iter: int = 6,
    eps: float = 1e-12,
) -> float | None:
    x = torch.randn(dim, device=device, dtype=dtype)
    x = x / (torch.linalg.norm(x) + eps)

    lam = None
    for _ in range(num_iter):
        y = A_times(x)
        if not torch.isfinite(y).all():
            return None

        y_norm = torch.linalg.norm(y)
        if y_norm <= eps:
            return None

        x = y / (y_norm + eps)
        Ax = A_times(x)
        if not torch.isfinite(Ax).all():
            return None

        lam = torch.dot(x, Ax).item()

    if lam is None or not np.isfinite(lam) or lam <= 0:
        return None
    return float(lam)


def ihvp(
    model: torch.nn.Module,
    loss: torch.Tensor,
    v: torch.Tensor,
    mu: float = 1.0,
    tol: float = 1e-6,
    max_iter: int = 200,
    verbose: bool = False,
) -> torch.Tensor:
    x = mu * v
    v_norm = torch.linalg.norm(v).item() + 1e-12

    for t in range(max_iter):
        r = v - hvp(model, loss, x)
        step = mu * r
        x_next = x + step

        rel_residual = torch.linalg.norm(r).item() / v_norm
        if verbose:
            print(
                f"[ihvp] iter={t + 1}/{max_iter} rel_residual={rel_residual:.3e}",
                end="\r",
                flush=True,
            )

        if not torch.isfinite(x_next).all():
            raise RuntimeError("ihvp failed: non-finite iterate.")

        if rel_residual < tol:
            if verbose:
                print()
            return x_next

        x = x_next

    if verbose:
        print()
    return x


def p_lissa(
    model: torch.nn.Module,
    loss: torch.Tensor,
    g_full: torch.Tensor,
    index_list,
    damping: float = 0.0,
    mu: float = 1.0,
    tol: float = 1e-6,
    max_iter: int = 200,
    max_restarts: int = 8,
    verbose: bool = False,
) -> torch.Tensor:
    full_dim = g_full.numel()
    idx = _as_index_tensor(index_list, g_full.device)
    rhs = _project_subset(hvp(model, loss, g_full), idx)

    def A_times(x_sub: torch.Tensor) -> torch.Tensor:
        x_full = _embed_subset(x_sub, idx, full_dim)
        return _project_subset(
            hvp(model, loss, hvp(model, loss, x_full)),
            idx,
        )

    return p_lissa_inverse(
        a_times=A_times,
        rhs=rhs,
        damping=damping,
        mu=mu,
        tol=tol,
        max_iter=max_iter,
        max_restarts=max_restarts,
        verbose=verbose,
    )


def p_lissa_inverse(
    a_times,
    rhs: torch.Tensor,
    damping: float = 0.0,
    mu: float = 1.0,
    tol: float = 1e-6,
    max_iter: int = 200,
    max_restarts: int = 8,
    power_iter_steps: int = 6,
    keep_best_iterate: bool = True,
    verbose: bool = False,
) -> torch.Tensor:
    STABLE_PATIENCE = 2
    BAD_PATIENCE = 2
    STALL_WINDOW = 8
    MIN_PROGRESS = 1e-3
    BLOWUP_FACTOR = 5.0
    X_BLOWUP = 1e4
    EPS = 1e-12

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

    rhs_norm = torch.linalg.norm(rhs).item() + EPS

    def apply_operator(x: torch.Tensor) -> torch.Tensor:
        out = a_times(x)
        if damping > 0:
            out = out + damping * x
        return out

    lam_max_hat = None
    if power_iter_steps > 0:
        lam_max_hat = _estimate_lmax_power(
            A_times=apply_operator,
            dim=rhs.numel(),
            device=rhs.device,
            dtype=rhs.dtype,
            num_iter=power_iter_steps,
        )
    if lam_max_hat is not None:
        mu = min(mu, 0.9 / max(lam_max_hat, EPS))

    best_x = None
    best_rel_residual = float("inf")

    for restart in range(max_restarts):
        x = rhs.clone()
        x.mul_(mu)
        x0_norm = torch.linalg.norm(x).item()
        ema_residual = None
        best_ema = float("inf")
        stable_hits = 0
        bad_hits = 0
        recent_ema = []

        for t in range(max_iter):
            Ax = apply_operator(x)
            r = rhs - Ax

            if (
                (not torch.isfinite(r).all())
                or (not torch.isfinite(x).all())
            ):
                bad_hits += 1
            else:
                residual_norm = torch.linalg.norm(r).item()
                rel_residual = residual_norm / rhs_norm
                step_norm = abs(mu) * residual_norm
                x_norm = torch.linalg.norm(x).item()

                if ema_residual is None:
                    ema_residual = residual_norm
                else:
                    ema_residual = 0.9 * ema_residual + 0.1 * residual_norm

                best_ema = min(best_ema, ema_residual)

                if rel_residual < best_rel_residual:
                    best_rel_residual = rel_residual
                    if keep_best_iterate:
                        best_x = x.detach().clone()

                residual_ok = rel_residual < tol
                step_ok = step_norm <= tol * max(x_norm, 1.0)

                if residual_ok and step_ok:
                    stable_hits += 1
                else:
                    stable_hits = 0

                if stable_hits >= STABLE_PATIENCE:
                    x.add_(r, alpha=mu)
                    if verbose:
                        print(
                            f"[p_lissa] converged at restart={restart}, "
                            f"iter={t + 1}, rel_residual={rel_residual:.3e}"
                        )
                    return x

                if ema_residual > BLOWUP_FACTOR * max(best_ema, 1e-30):
                    bad_hits += 1
                else:
                    bad_hits = 0

                if x_norm > X_BLOWUP * max(x0_norm, rhs_norm, 1.0):
                    bad_hits += 1

                if bad_hits >= BAD_PATIENCE:
                    mu *= 0.5
                    if verbose:
                        print(
                            f"[p_lissa] divergence at restart={restart}, "
                            f"iter={t + 1}, new_mu={mu:.3e}"
                        )
                    break

                recent_ema.append(ema_residual)
                if len(recent_ema) > STALL_WINDOW:
                    recent_ema.pop(0)

                if len(recent_ema) == STALL_WINDOW:
                    progress = (recent_ema[0] - recent_ema[-1]) / max(
                        recent_ema[0], 1e-30
                    )
                    if progress < MIN_PROGRESS:
                        mu *= 0.5
                        if verbose:
                            print(
                                f"[p_lissa] stagnation at restart={restart}, "
                                f"iter={t + 1}, new_mu={mu:.3e}"
                            )
                        break

                if verbose:
                    print(
                        f"[p_lissa] restart={restart} iter={t + 1}/{max_iter} "
                        f"mu={mu:.3e} rel_residual={rel_residual:.3e}",
                        end="\r",
                        flush=True,
                    )

            if torch.isfinite(r).all() and torch.isfinite(x).all():
                x.add_(r, alpha=mu)
        else:
            if verbose:
                print()
            return x

    if best_x is not None:
        if verbose:
            print(
                f"\n[p_lissa] all restarts exhausted. "
                f"Returning best iterate with rel_residual={best_rel_residual:.3e}"
            )
        return best_x

    if torch.isfinite(x).all():
        if verbose:
            print(
                "\n[p_lissa] all restarts exhausted. Returning last finite iterate."
            )
        return x

    raise RuntimeError("p_lissa failed: no finite iterate was produced.")


def iphvp(
    model: torch.nn.Module,
    loss: torch.Tensor,
    v: torch.Tensor,
    index_list: np.ndarray,
    mu: float = 1.0,
    tol: float = 1e-6,
    max_iter: int = 200,
    max_restarts: int = 8,
    verbose: bool = False,
) -> torch.Tensor:
    full_v = v
    if full_v.numel() == len(index_list):
        full_dim = sum(parameter.numel() for parameter in model.parameters())
        full_v = _embed_subset(full_v, index_list, full_dim)

    return p_lissa(
        model=model,
        loss=loss,
        g_full=full_v,
        index_list=index_list,
        damping=0.0,
        mu=mu,
        tol=tol,
        max_iter=max_iter,
        max_restarts=max_restarts,
        verbose=verbose,
    )
