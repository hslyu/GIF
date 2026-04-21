import numpy as np
import torch

from gif.influence import CGInfluence, cg_update, compute_gradient
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


def test_cg_update_matches_dense_oracle_with_damping():
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

    update = cg_update(
        model=model,
        total_loss=total_loss,
        target_loss=target_loss,
        index_list=index_list,
        damping=damping,
        tol=1e-12,
        max_iter=20,
    )

    torch.testing.assert_close(update, oracle, atol=1e-10, rtol=1e-10)


def test_cg_class_api_returns_details():
    model, total_loss, target_loss = _toy_linear_problem()
    index_list = np.array([0, 2], dtype=int)

    result = CGInfluence().compute(
        model=model,
        total_loss=total_loss,
        target_loss=target_loss,
        index_list=index_list,
        damping=0.1,
        tol=1e-12,
        max_iter=20,
        return_details=True,
    )

    assert "update" in result
    assert "details" in result
    assert torch.isfinite(result["update"]).all()
