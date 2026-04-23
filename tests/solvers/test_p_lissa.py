import numpy as np
import torch

from gif.influence import _embed_subset, _project_subset, compute_gradient, hvp
from gif.solvers import p_lissa


def _build_restricted_system(model, total_loss, index_list, g_full):
    idx = np.asarray(index_list, dtype=int)
    full_dim = g_full.numel()

    def a_times(x_sub: torch.Tensor) -> torch.Tensor:
        x_full = _embed_subset(x_sub, idx, full_dim)
        return _project_subset(
            hvp(model, total_loss, hvp(model, total_loss, x_full)),
            idx,
        )

    rhs = _project_subset(hvp(model, total_loss, g_full), idx)
    basis = torch.eye(len(idx), dtype=g_full.dtype, device=g_full.device)
    matrix = torch.stack([a_times(basis[i]) for i in range(len(idx))], dim=1)
    return matrix, rhs, a_times


def _toy_linear_problem():
    model = torch.nn.Linear(2, 1, bias=True).double()
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.3, -0.2]], dtype=torch.float64))
        model.bias.copy_(torch.tensor([0.05], dtype=torch.float64))

    inputs = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, -1.0],
        ],
        dtype=torch.float64,
    )
    targets = torch.tensor([[0.4], [-0.1], [0.2], [0.9]], dtype=torch.float64)
    criterion = torch.nn.MSELoss()
    total_loss = criterion(model(inputs), targets)
    target_loss = criterion(model(inputs[:1]), targets[:1])
    return model, total_loss, target_loss


def test_p_lissa_matches_oracle_restricted_solution():
    model, total_loss, target_loss = _toy_linear_problem()
    g_full = compute_gradient(model, target_loss)
    index_list = np.array([0, 2], dtype=int)
    matrix, rhs, _ = _build_restricted_system(model, total_loss, index_list, g_full)

    oracle = torch.linalg.solve(matrix, rhs)
    approx = p_lissa(
        model=model,
        loss=total_loss,
        g_full=g_full,
        index_list=index_list,
        mu=0.25,
        tol=1e-12,
        max_iter=200,
        max_restarts=4,
    )

    torch.testing.assert_close(approx, oracle, atol=5e-5, rtol=1e-4)


def test_p_lissa_matches_damped_oracle_restricted_solution():
    model, total_loss, target_loss = _toy_linear_problem()
    g_full = compute_gradient(model, target_loss)
    index_list = np.array([0, 2], dtype=int)
    matrix, rhs, _ = _build_restricted_system(model, total_loss, index_list, g_full)
    damping = 1e-2

    oracle = torch.linalg.solve(
        matrix + damping * torch.eye(matrix.size(0), dtype=matrix.dtype), rhs
    )
    approx = p_lissa(
        model=model,
        loss=total_loss,
        g_full=g_full,
        index_list=index_list,
        damping=damping,
        mu=0.25,
        tol=1e-12,
        max_iter=200,
        max_restarts=4,
    )

    torch.testing.assert_close(approx, oracle, atol=5e-5, rtol=1e-4)


def test_p_lissa_conservative_mu_reduces_residual():
    model, total_loss, target_loss = _toy_linear_problem()
    g_full = compute_gradient(model, target_loss)
    index_list = np.array([0, 2], dtype=int)
    matrix, rhs, _ = _build_restricted_system(model, total_loss, index_list, g_full)

    short = p_lissa(
        model=model,
        loss=total_loss,
        g_full=g_full,
        index_list=index_list,
        mu=0.05,
        tol=1e-12,
        max_iter=2,
        max_restarts=1,
    )
    long = p_lissa(
        model=model,
        loss=total_loss,
        g_full=g_full,
        index_list=index_list,
        mu=0.05,
        tol=1e-12,
        max_iter=80,
        max_restarts=1,
    )

    short_residual = torch.linalg.norm(matrix @ short - rhs).item()
    long_residual = torch.linalg.norm(matrix @ long - rhs).item()

    assert long_residual < short_residual


def test_p_lissa_large_mu_triggers_restart_or_shrinking(capsys, monkeypatch):
    model = torch.nn.Linear(1, 1, bias=False).double()
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0]], dtype=torch.float64))

    inputs = torch.tensor([[10.0]], dtype=torch.float64)
    targets = torch.tensor([[0.0]], dtype=torch.float64)
    criterion = torch.nn.MSELoss()
    total_loss = criterion(model(inputs), targets)
    target_loss = criterion(model(inputs), targets)
    g_full = compute_gradient(model, target_loss)
    index_list = np.array([0], dtype=int)

    monkeypatch.setattr(
        "gif.solvers.iterative._estimate_lmax_power", lambda *args, **kwargs: None
    )

    approx = p_lissa(
        model=model,
        loss=total_loss,
        g_full=g_full,
        index_list=index_list,
        mu=1.0,
        tol=1e-12,
        max_iter=2,
        max_restarts=3,
        verbose=True,
    )
    output = capsys.readouterr().out

    assert torch.isfinite(approx).all()
    assert "divergence" in output or "stagnation" in output or "new_mu=" in output


def test_p_lissa_survives_flat_direction_stress():
    model = torch.nn.Linear(2, 1, bias=False).double()
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.7, -0.4]], dtype=torch.float64))

    inputs = torch.tensor(
        [
            [1.0, 1.0e-4],
            [2.0, 2.1e-4],
            [3.0, 3.1e-4],
            [4.0, 4.05e-4],
        ],
        dtype=torch.float64,
    )
    targets = torch.tensor([[0.3], [0.6], [0.9], [1.2]], dtype=torch.float64)
    criterion = torch.nn.MSELoss()
    total_loss = criterion(model(inputs), targets)
    target_loss = criterion(model(inputs[:1]), targets[:1])
    g_full = compute_gradient(model, target_loss)

    full_index_list = np.array([0, 1], dtype=int)
    restricted_index_list = np.array([0], dtype=int)

    full_matrix, full_rhs, _ = _build_restricted_system(
        model, total_loss, full_index_list, g_full
    )
    restricted_matrix, restricted_rhs, _ = _build_restricted_system(
        model, total_loss, restricted_index_list, g_full
    )

    cond = torch.linalg.cond(full_matrix)
    assert cond > 1e8

    full_approx = p_lissa(
        model=model,
        loss=total_loss,
        g_full=g_full,
        index_list=full_index_list,
        mu=0.1,
        tol=1e-12,
        max_iter=50,
        max_restarts=3,
    )
    restricted_approx = p_lissa(
        model=model,
        loss=total_loss,
        g_full=g_full,
        index_list=restricted_index_list,
        mu=0.1,
        tol=1e-12,
        max_iter=50,
        max_restarts=3,
    )

    assert torch.isfinite(full_approx).all()
    assert torch.isfinite(restricted_approx).all()

    full_residual = torch.linalg.norm(full_matrix @ full_approx - full_rhs).item()
    restricted_residual = torch.linalg.norm(
        restricted_matrix @ restricted_approx - restricted_rhs
    ).item()

    assert restricted_residual <= full_residual
