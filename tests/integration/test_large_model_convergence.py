import os

import numpy as np
import pytest
import torch

from gif.influence import _embed_subset, _project_subset, compute_gradient, generalized_influence, hvp
from gif.models import FullyConnectedNet


@pytest.mark.skipif(
    os.getenv("GIF_RUN_LARGE_MODEL_TESTS") != "1",
    reason="Set GIF_RUN_LARGE_MODEL_TESTS=1 to run the large-model convergence test.",
)
def test_large_fully_connected_model_residual_decreases_with_more_iterations():
    torch.manual_seed(0)

    model = FullyConnectedNet(
        input_size=64,
        hidden_size=256,
        output_size=10,
        num_layers=6,
        dropout_prob=0.0,
    ).double()
    model.eval()

    inputs = torch.randn(8, 64, dtype=torch.float64)
    targets = torch.randint(0, 10, (8,), dtype=torch.long)
    criterion = torch.nn.CrossEntropyLoss()

    total_loss = criterion(model(inputs), targets)
    target_loss = criterion(model(inputs[:2]), targets[:2])

    g_full = compute_gradient(model, target_loss)
    index_list = np.arange(min(128, g_full.numel()), dtype=int)
    idx = torch.as_tensor(index_list, dtype=torch.long)
    full_dim = g_full.numel()

    def a_times(x_sub: torch.Tensor) -> torch.Tensor:
        x_full = _embed_subset(x_sub, idx, full_dim)
        return _project_subset(
            hvp(model, total_loss, hvp(model, total_loss, x_full)),
            idx,
        )

    rhs = _project_subset(hvp(model, total_loss, g_full), idx)
    rhs_norm = torch.linalg.norm(rhs).item() + 1e-12

    short = generalized_influence(
        model=model,
        total_loss=total_loss,
        target_loss=target_loss,
        index_list=index_list,
        mu=1.0,
        tol=1e-10,
        max_iter=5,
        max_restarts=4,
    )
    long = generalized_influence(
        model=model,
        total_loss=total_loss,
        target_loss=target_loss,
        index_list=index_list,
        mu=1.0,
        tol=1e-10,
        max_iter=200,
        max_restarts=4,
    )

    short_residual = torch.linalg.norm(a_times(short) - rhs).item() / rhs_norm
    long_residual = torch.linalg.norm(a_times(long) - rhs).item() / rhs_norm

    assert torch.isfinite(short).all()
    assert torch.isfinite(long).all()
    assert long_residual < short_residual
    assert long_residual < 0.5
