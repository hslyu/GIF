import torch
from torch import nn

from gif.influence import EKFACInfluence, ekfac_update, kfac_update
from gif.models import TextTransformerClassifier


def test_ekfac_update_returns_full_vector_for_fcn():
    model = nn.Sequential(
        nn.Linear(4, 6),
        nn.ReLU(),
        nn.Linear(6, 3),
    ).double()
    criterion = nn.CrossEntropyLoss()
    retained_inputs = torch.randn(8, 4, dtype=torch.float64)
    retained_targets = torch.tensor([0, 1, 2, 1, 0, 2, 1, 0], dtype=torch.long)
    target_inputs = retained_inputs[:3]
    target_targets = retained_targets[:3]

    update = ekfac_update(
        model=model,
        retained_inputs=retained_inputs,
        retained_targets=retained_targets,
        target_inputs=target_inputs,
        target_targets=target_targets,
        criterion=criterion,
        damping=1e-3,
    )

    expected_dim = sum(parameter.numel() for parameter in model.parameters())
    assert update.ndim == 1
    assert update.shape[0] == expected_dim
    assert torch.isfinite(update).all()


def test_ekfac_update_returns_full_vector_for_cnn():
    model = nn.Sequential(
        nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(4 * 8 * 8, 2),
    ).double()
    criterion = nn.CrossEntropyLoss()
    retained_inputs = torch.randn(6, 1, 8, 8, dtype=torch.float64)
    retained_targets = torch.tensor([0, 1, 0, 1, 1, 0], dtype=torch.long)
    target_inputs = retained_inputs[:2]
    target_targets = retained_targets[:2]

    result = EKFACInfluence().compute(
        model=model,
        retained_inputs=retained_inputs,
        retained_targets=retained_targets,
        target_inputs=target_inputs,
        target_targets=target_targets,
        criterion=criterion,
        damping=1e-3,
        return_details=True,
    )

    expected_dim = sum(parameter.numel() for parameter in model.parameters())
    assert result["update"].shape[0] == expected_dim
    assert result["details"]["modules"]["0"]["corrected_diagonal_shape"] == [4, 10]
    assert torch.isfinite(result["update"]).all()


def test_ekfac_update_supports_attention_projection_layers():
    torch.manual_seed(0)
    model = TextTransformerClassifier(
        vocab_size=32,
        max_len=8,
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        num_classes=3,
        dropout_prob=0.0,
    ).double()
    criterion = nn.CrossEntropyLoss()
    retained_ids = torch.randint(1, 32, (6, 8), dtype=torch.long)
    retained_mask = torch.ones(6, 8, dtype=torch.long)
    retained_targets = torch.tensor([0, 1, 2, 1, 0, 2], dtype=torch.long)
    target_ids = retained_ids[:2]
    target_mask = retained_mask[:2]
    target_targets = retained_targets[:2]

    result = ekfac_update(
        model=model,
        retained_inputs=(retained_ids, retained_mask),
        retained_targets=retained_targets,
        target_inputs=(target_ids, target_mask),
        target_targets=target_targets,
        criterion=criterion,
        damping=1e-3,
        return_details=True,
    )

    expected_dim = sum(parameter.numel() for parameter in model.parameters())
    assert result["update"].shape[0] == expected_dim
    assert "layers.0.q_proj" in result["details"]["modules"]
    assert result["details"]["modules"]["layers.0.q_proj"]["role"] == "attention_projection"
    assert result["details"]["modules"]["layers.0.q_proj"]["corrected_diagonal_shape"] == [16, 17]
    assert torch.isfinite(result["update"]).all()


def test_ekfac_and_kfac_produce_distinct_updates():
    model = nn.Sequential(
        nn.Linear(5, 7),
        nn.Tanh(),
        nn.Linear(7, 4),
    ).double()
    criterion = nn.CrossEntropyLoss()
    retained_inputs = torch.randn(10, 5, dtype=torch.float64)
    retained_targets = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 1, 0], dtype=torch.long)
    target_inputs = retained_inputs[:4]
    target_targets = retained_targets[:4]

    kfac = kfac_update(
        model=model,
        retained_inputs=retained_inputs,
        retained_targets=retained_targets,
        target_inputs=target_inputs,
        target_targets=target_targets,
        criterion=criterion,
        damping=1e-3,
    )
    ekfac = ekfac_update(
        model=model,
        retained_inputs=retained_inputs,
        retained_targets=retained_targets,
        target_inputs=target_inputs,
        target_targets=target_targets,
        criterion=criterion,
        damping=1e-3,
    )

    assert torch.isfinite(kfac).all()
    assert torch.isfinite(ekfac).all()
    assert not torch.allclose(kfac, ekfac)
