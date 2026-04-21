from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_ROOT = Path("/home/hslyu/research/rework/GIF/scripts/search")
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from _mnist_unlearning_common import compute_method_update  # noqa: E402


def _build_toy_setup():
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(4, 6),
        nn.ReLU(),
        nn.Linear(6, 2),
    )
    criterion = nn.CrossEntropyLoss()

    train_inputs = torch.randn(8, 1, 2, 2)
    train_targets = torch.tensor([0, 1, 0, 1, 1, 0, 1, 1])
    retained_mask = train_targets != 0
    retained_inputs = train_inputs[retained_mask]
    retained_targets = train_targets[retained_mask]
    sampled_inputs = train_inputs[~retained_mask][:2]
    sampled_targets = train_targets[~retained_mask][:2]
    train_loader = DataLoader(
        TensorDataset(train_inputs, train_targets), batch_size=4, shuffle=False
    )
    return (
        model,
        criterion,
        train_loader,
        sampled_inputs,
        sampled_targets,
        retained_inputs,
        retained_targets,
    )


def test_additional_influence_schemes_produce_finite_normalized_updates():
    device = torch.device("cpu")
    for scheme in ("influence", "second_influence", "freeze_influence"):
        (
            model,
            criterion,
            train_loader,
            sampled_inputs,
            sampled_targets,
            retained_inputs,
            retained_targets,
        ) = _build_toy_setup()

        selector, normalized_update = compute_method_update(
            scheme=scheme,
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            sampled_inputs=sampled_inputs,
            sampled_targets=sampled_targets,
            all_target_count=3,
            retained_inputs=retained_inputs,
            retained_targets=retained_targets,
            param_ratio=0.5,
            caps_lam=1e-5,
            caps_min_curv=1e-12,
            batch_size=4,
            tol=1e-4,
            mu=1.0,
            max_iter=5,
            device=device,
        )

        assert len(selector.get_parameters()) > 0
        assert normalized_update.ndim == 1
        assert torch.isfinite(normalized_update).all()
        assert torch.isclose(
            torch.linalg.norm(normalized_update), torch.tensor(1.0), atol=1e-4
        )
