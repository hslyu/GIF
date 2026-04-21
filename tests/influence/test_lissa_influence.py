import numpy as np
import torch

from gif.influence import LiSSAInfluence, compute_gradient, lissa_update
from gif.influence.restricted import build_restricted_system


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


def test_lissa_update_matches_dense_oracle_with_damping():
    model, total_loss, target_loss = _toy_linear_problem()
    index_list = np.array([0, 2], dtype=int)
    damping = 0.1
    g_full = compute_gradient(model, target_loss)
    rhs, a_times = build_restricted_system(model, total_loss, g_full, index_list)
    basis = torch.eye(len(index_list), dtype=rhs.dtype, device=rhs.device)
    matrix = torch.stack([a_times(basis[i]) for i in range(len(index_list))], dim=1)
    oracle = torch.linalg.solve(
        matrix + damping * torch.eye(len(index_list), dtype=rhs.dtype, device=rhs.device),
        rhs,
    )

    update = lissa_update(
        model=model,
        total_loss=total_loss,
        target_loss=target_loss,
        index_list=index_list,
        damping=damping,
        mu=0.25,
        tol=1e-12,
        max_iter=200,
        max_restarts=4,
    )

    torch.testing.assert_close(update, oracle, atol=5e-5, rtol=1e-4)


def test_lissa_class_api_returns_details():
    model, total_loss, target_loss = _toy_linear_problem()
    index_list = np.array([0, 2], dtype=int)

    result = LiSSAInfluence().compute(
        model=model,
        total_loss=total_loss,
        target_loss=target_loss,
        index_list=index_list,
        damping=0.1,
        mu=0.25,
        tol=1e-12,
        max_iter=200,
        max_restarts=4,
        return_details=True,
    )

    assert "update" in result
    assert "details" in result
    assert torch.isfinite(result["update"]).all()
