import os
from pathlib import Path

import numpy as np
import pytest
import torch

from gif.influence import _embed_subset, _project_subset, compute_gradient, generalized_influence, hvp
from gif.models import ResNet18


CHECKPOINT_PATH = Path(__file__).resolve().parents[2] / "checkpoints" / "mnist_resnet18.pth"


@pytest.mark.skipif(
    os.getenv("GIF_RUN_RESNET18_TESTS") != "1",
    reason="Set GIF_RUN_RESNET18_TESTS=1 to run the ResNet18 convergence test.",
)
def test_resnet18_generalized_influence_is_stable():
    if not CHECKPOINT_PATH.is_file():
        pytest.skip(f"Missing checkpoint: {CHECKPOINT_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    model = ResNet18(in_channels=1).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["net"])
    model.eval()

    inputs = torch.randn(8, 1, 32, 32, device=device)
    targets = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device=device, dtype=torch.long)

    retain_inputs = inputs[:6]
    retain_targets = targets[:6]
    target_inputs = inputs[6:]
    target_targets = targets[6:]

    criterion = torch.nn.CrossEntropyLoss()
    total_loss = criterion(model(retain_inputs), retain_targets)
    target_loss = criterion(model(target_inputs), target_targets)

    g_full = compute_gradient(model, target_loss)
    total_params = g_full.numel()
    index_list = np.arange(min(256, total_params), dtype=int)
    idx = torch.as_tensor(index_list, device=device, dtype=torch.long)

    full_dim = total_params

    def a_times(x_sub: torch.Tensor) -> torch.Tensor:
        x_full = _embed_subset(x_sub, idx, full_dim)
        return _project_subset(
            hvp(model, total_loss, hvp(model, total_loss, x_full)),
            idx,
        )

    rhs = _project_subset(hvp(model, total_loss, g_full), idx)
    gif_sub = generalized_influence(
        model=model,
        total_loss=total_loss,
        target_loss=target_loss,
        index_list=index_list,
        mu=0.5,
        tol=1e-10,
        max_iter=200,
        max_restarts=4,
    )

    residual = torch.linalg.norm(a_times(gif_sub) - rhs).item()
    rhs_norm = torch.linalg.norm(rhs).item() + 1e-12

    assert torch.isfinite(gif_sub).all()
    assert residual / rhs_norm < 1e-2
