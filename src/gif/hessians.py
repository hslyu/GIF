#!/usr/bin/env python3
import gc

import numpy as np
import torch


def compute_gradient(
    model: torch.nn.Module,
    loss: torch.Tensor,
    create_graph: bool = False,
) -> torch.Tensor:
    grads = torch.autograd.grad(
        loss,
        list(model.parameters()),
        retain_graph=True,
        create_graph=create_graph,
    )
    return torch.cat([g.contiguous().view(-1) for g in grads])


def compute_hessian(model: torch.nn.Module, loss: torch.Tensor) -> torch.Tensor:
    gradients = compute_gradient(model, loss, create_graph=True)
    hessian = torch.zeros(
        gradients.numel(),
        gradients.numel(),
        device=gradients.device,
        dtype=gradients.dtype,
    )
    for idx in range(gradients.numel()):
        second_gradients = torch.autograd.grad(
            gradients[idx],
            list(model.parameters()),
            retain_graph=True,
        )
        hessian[idx] = torch.cat([grad.contiguous().view(-1) for grad in second_gradients])
    return hessian


def hvp(
    model: torch.nn.Module,
    loss: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """
    Hessian-vector product: H v
    """
    grads = torch.autograd.grad(
        loss,
        list(model.parameters()),
        create_graph=True,
        retain_graph=True,
    )
    flat_grads = torch.cat([g.contiguous().view(-1) for g in grads])
    hv = torch.autograd.grad(
        flat_grads,
        list(model.parameters()),
        grad_outputs=v,
        retain_graph=True,
    )
    return torch.cat([g.contiguous().view(-1) for g in hv])


def _as_index_tensor(index_list, device):
    if isinstance(index_list, torch.Tensor):
        return index_list.to(device=device, dtype=torch.long)
    return torch.as_tensor(index_list, device=device, dtype=torch.long)


def _embed_subset(
    v_sub: torch.Tensor,
    index_list,
    full_dim: int,
) -> torch.Tensor:
    idx = _as_index_tensor(index_list, v_sub.device)
    out = torch.zeros(full_dim, device=v_sub.device, dtype=v_sub.dtype)
    out[idx] = v_sub
    return out


def _project_subset(
    v_full: torch.Tensor,
    index_list,
) -> torch.Tensor:
    idx = _as_index_tensor(index_list, v_full.device)
    return v_full.index_select(0, idx)


def _resolve_mu(mu: float | None = None, normalizer: float | None = None) -> float:
    if normalizer is not None:
        if normalizer <= 0:
            raise ValueError("normalizer must be positive")
        return 1.0 / float(normalizer)
    if mu is not None:
        return float(mu)
    return 1.0


def lissa_ihvp(
    model: torch.nn.Module,
    loss: torch.Tensor,
    v: torch.Tensor,
    mu: float = 1.0,
    tol: float = 1e-8,
    max_iter: int = 200,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Classical LiSSA / Neumann solver for H^{-1} v.
    This is for the FULL-parameter IF path.
    """
    rhs = mu * v
    I = rhs.clone()

    for t in range(max_iter):
        HI = hvp(model, loss, I)
        I_next = rhs + I - mu * HI
        diff = torch.norm(I_next - I)

        if verbose:
            print(f"LiSSA [{t + 1}/{max_iter}] diff={diff.item():.3e}", end="\r")

        if not torch.isfinite(diff):
            raise RuntimeError("LiSSA diverged (non-finite iterate).")

        if diff < tol:
            if verbose:
                print()
            return I_next

        I = I_next

    if verbose:
        print()
    return I


def p_lissa(
    model: torch.nn.Module,
    loss: torch.Tensor,
    g_full: torch.Tensor,
    index_list,
    mu: float = 1.0,
    tol: float = 1e-8,
    max_iter: int = 200,
    max_restarts: int = 10,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Paper-faithful p-LiSSA for GIF:
        I_t = mu * H_J^T g + (I - mu * H_J^T H_J) I_{t-1}

    Returns the J-subspace solution H_J^+ g (up to your sign convention).
    """
    full_dim = g_full.numel()
    idx = _as_index_tensor(index_list, g_full.device)

    # rhs = H_J^T g   (select J coords after H g, using symmetry of H)
    rhs = _project_subset(hvp(model, loss, g_full), idx)

    for restart in range(max_restarts):
        I = mu * rhs.clone()
        prev_diff = None

        for t in range(max_iter):
            I_full = _embed_subset(I, idx, full_dim)

            # H_J^T H_J I:
            # 1) embed J-vector into full space
            # 2) apply H once -> H_J I in full coordinates
            # 3) apply H again and select J coords -> H_J^T H_J I
            gram_I = _project_subset(
                hvp(model, loss, hvp(model, loss, I_full)),
                idx,
            )

            I_next = mu * rhs + I - mu * gram_I
            diff = torch.norm(I_next - I)

            if verbose:
                print(
                    f"p-LiSSA restart={restart} iter={t + 1}/{max_iter} "
                    f"mu={mu:.3e} diff={diff.item():.3e}",
                    end="\r",
                )

            # Divergence / instability -> halve mu and restart from scratch
            if (not torch.isfinite(diff)) or (
                prev_diff is not None and diff > prev_diff
            ):
                mu *= 0.5
                break

            if diff < tol:
                if verbose:
                    print()
                return I_next

            I = I_next
            prev_diff = diff
        else:
            # loop ended without break
            if verbose:
                print()
            return I

    raise RuntimeError("p-LiSSA failed to converge after all restarts.")


def influence(
    model: torch.nn.Module,
    total_loss: torch.Tensor,
    loss: torch.Tensor,
    mu: float = 1.0,
    tol: float = 1e-8,
    max_iter: int = 200,
    verbose: bool = False,
    return_negative_if: bool = False,
    step: float | None = None,
    normalizer: float | None = None,
) -> torch.Tensor:
    """
    Full-parameter IF path.

    Paper notation defines IF as -H^{-1} g_w.
    Many codebases instead return the unlearning displacement +H^{-1} g_w for epsilon=-1.
    Pick one convention and keep it consistent downstream.
    """
    g = compute_gradient(model, loss)
    sol = lissa_ihvp(
        model=model,
        loss=total_loss,
        v=g,
        mu=_resolve_mu(mu=mu, normalizer=normalizer),
        tol=tol,
        max_iter=max_iter,
        verbose=verbose,
    )
    return -sol if return_negative_if else sol


def ihvp(
    model: torch.nn.Module,
    loss: torch.Tensor,
    v: torch.Tensor,
    tol: float = 1e-8,
    max_iter: int = 200,
    verbose: bool = False,
    mu: float = 1.0,
    step: float | None = None,
    normalizer: float | None = None,
) -> torch.Tensor:
    return lissa_ihvp(
        model=model,
        loss=loss,
        v=v,
        mu=_resolve_mu(mu=mu, normalizer=normalizer),
        tol=tol,
        max_iter=max_iter,
        verbose=verbose,
    )


def iphvp(
    model: torch.nn.Module,
    loss: torch.Tensor,
    v: torch.Tensor,
    index_list: np.ndarray,
    tol: float = 1e-8,
    max_iter: int = 200,
    verbose: bool = False,
    mu: float = 1.0,
    max_restarts: int = 10,
    step: float | None = None,
    normalizer: float | None = None,
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
        mu=_resolve_mu(mu=mu, normalizer=normalizer),
        tol=tol,
        max_iter=max_iter,
        max_restarts=max_restarts,
        verbose=verbose,
    )


def generalized_influence(
    model: torch.nn.Module,
    total_loss: torch.Tensor,
    target_loss: torch.Tensor,
    index_list: np.ndarray,
    mu: float = 1.0,
    tol: float = 1e-8,
    max_iter: int = 200,
    max_restarts: int = 10,
    verbose: bool = False,
    return_full_vector: bool = False,
    return_negative_if: bool = False,
    step: float | None = None,
    normalizer: float | None = None,
) -> torch.Tensor:
    """
    GIF / p-LiSSA path.

    Returns the J-subspace vector by default.
    If return_full_vector=True, scatters it back into the full parameter space.
    """
    g_full = compute_gradient(model, target_loss)

    gif_sub = p_lissa(
        model=model,
        loss=total_loss,
        g_full=g_full,
        index_list=index_list,
        mu=_resolve_mu(mu=mu, normalizer=normalizer),
        tol=tol,
        max_iter=max_iter,
        max_restarts=max_restarts,
        verbose=verbose,
    )

    if return_negative_if:
        gif_sub = -gif_sub

    if return_full_vector:
        return _embed_subset(gif_sub, index_list, g_full.numel())

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return gif_sub
