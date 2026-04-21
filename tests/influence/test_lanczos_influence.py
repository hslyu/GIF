import numpy as np
import torch

from gif.influence import LanczosInfluence, compute_gradient, lanczos_update
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


def test_lanczos_update_matches_dense_oracle_with_full_rank():
    model, total_loss, target_loss = _toy_linear_problem()
    index_list = np.array([0, 2], dtype=int)
    g_full = compute_gradient(model, target_loss)
    rhs, a_times = build_restricted_system(model, total_loss, g_full, index_list)
    matrix = torch.stack(
        [a_times(torch.eye(len(index_list), dtype=rhs.dtype)[i]) for i in range(len(index_list))],
        dim=1,
    )
    oracle = torch.linalg.solve(matrix, rhs)

    update = lanczos_update(
        model=model,
        total_loss=total_loss,
        target_loss=target_loss,
        index_list=index_list,
        rank=len(index_list),
        tol=1e-12,
        max_iter=20,
    )

    torch.testing.assert_close(update, oracle, atol=1e-10, rtol=1e-10)


def test_lanczos_class_api_returns_details():
    model, total_loss, target_loss = _toy_linear_problem()
    index_list = np.array([0, 2], dtype=int)

    result = LanczosInfluence().compute(
        model=model,
        total_loss=total_loss,
        target_loss=target_loss,
        index_list=index_list,
        rank=len(index_list),
        damping=1e-4,
        tol=1e-12,
        max_iter=20,
        return_details=True,
    )

    assert "update" in result
    assert "details" in result
    assert torch.isfinite(result["update"]).all()
    assert result["details"]["rank"] == len(index_list)
