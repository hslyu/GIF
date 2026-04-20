import numpy as np
import torch

from gif.influence import (
    _embed_subset,
    _project_subset,
    compute_gradient,
    generalized_influence,
    hvp,
)
from gif.models import FullyConnectedNet
from gif.solvers import p_lissa


def _restricted_system(model, total_loss, index_list, g_full):
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
    return matrix, rhs


def test_generalized_influence_matches_p_lissa():
    model = torch.nn.Linear(2, 1, bias=True).double()
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.25, -0.15]], dtype=torch.float64))
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
    index_list = np.array([0, 2], dtype=int)

    g_full = compute_gradient(model, target_loss)
    direct = p_lissa(
        model=model,
        loss=total_loss,
        g_full=g_full,
        index_list=index_list,
        mu=0.25,
        tol=1e-12,
        max_iter=200,
        max_restarts=4,
    )
    api = generalized_influence(
        model=model,
        total_loss=total_loss,
        target_loss=target_loss,
        index_list=index_list,
        mu=0.25,
        tol=1e-12,
        max_iter=200,
        max_restarts=4,
    )

    torch.testing.assert_close(api, direct, atol=1e-6, rtol=1e-6)


def test_restricted_update_moves_target_loss_upward():
    torch.manual_seed(0)
    model = FullyConnectedNet(
        input_size=4,
        hidden_size=8,
        output_size=3,
        num_layers=3,
        dropout_prob=0.0,
    ).double()

    retain_inputs = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    retain_targets = torch.tensor([1, 1, 2, 2], dtype=torch.long)
    target_inputs = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float64,
    )
    target_targets = torch.tensor([0, 0], dtype=torch.long)

    criterion = torch.nn.CrossEntropyLoss()
    total_loss = criterion(model(retain_inputs), retain_targets)
    target_loss = criterion(model(target_inputs), target_targets)

    g_full = compute_gradient(model, target_loss)
    total_params = g_full.numel()
    index_list = np.arange(min(24, total_params), dtype=int)

    restricted_update = generalized_influence(
        model=model,
        total_loss=total_loss,
        target_loss=target_loss,
        index_list=index_list,
        mu=0.1,
        tol=1e-12,
        max_iter=120,
        max_restarts=4,
    )

    full_update = _embed_subset(restricted_update, index_list, total_params)
    target_direction = torch.dot(g_full, full_update).item()
    assert target_direction > 0

    before_target = target_loss.item()
    before_retain = total_loss.item()

    with torch.no_grad():
        flat_params = torch.nn.utils.parameters_to_vector(model.parameters())
        flat_params = flat_params + 1e-4 * full_update
        torch.nn.utils.vector_to_parameters(flat_params, model.parameters())

    after_target = criterion(model(target_inputs), target_targets).item()
    after_retain = criterion(model(retain_inputs), retain_targets).item()

    assert after_target > before_target
    assert abs(after_retain - before_retain) < 1e-2
