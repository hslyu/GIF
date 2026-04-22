import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from gif.influence.kfac import KFACFactorCollector
from gif.models import TextTransformerClassifier


def test_kfac_collector_gathers_linear_factors_for_fcn():
    model = nn.Sequential(
        nn.Linear(4, 6),
        nn.ReLU(),
        nn.Linear(6, 3),
    ).double()
    criterion = nn.CrossEntropyLoss()
    inputs = torch.randn(8, 4, dtype=torch.float64)
    targets = torch.tensor([0, 1, 2, 1, 0, 2, 1, 0], dtype=torch.long)

    collector = KFACFactorCollector(model)
    stats = collector.accumulate_batch(criterion, inputs, targets)

    assert len(stats) == 2
    first = stats["0"]
    second = stats["2"]
    assert first.info.kind == "linear"
    assert first.info.role == "linear"
    assert first.activation_cov.shape == (5, 5)
    assert first.gradient_cov.shape == (6, 6)
    assert second.activation_cov.shape == (7, 7)
    assert second.gradient_cov.shape == (3, 3)


def test_kfac_collector_gathers_conv2d_factors_for_cnn():
    model = nn.Sequential(
        nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(4 * 8 * 8, 2),
    ).double()
    criterion = nn.CrossEntropyLoss()
    inputs = torch.randn(6, 1, 8, 8, dtype=torch.float64)
    targets = torch.tensor([0, 1, 0, 1, 1, 0], dtype=torch.long)

    collector = KFACFactorCollector(model)
    stats = collector.accumulate_batch(criterion, inputs, targets)

    conv_stats = stats["0"]
    linear_stats = stats["3"]
    assert conv_stats.info.kind == "conv2d"
    assert conv_stats.activation_cov.shape == (10, 10)
    assert conv_stats.gradient_cov.shape == (4, 4)
    assert linear_stats.info.kind == "linear"
    assert linear_stats.activation_cov.shape == (257, 257)
    assert linear_stats.gradient_cov.shape == (2, 2)


def test_kfac_collector_marks_attention_projection_layers():
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
    input_ids = torch.randint(1, 32, (4, 8), dtype=torch.long)
    attention_mask = torch.ones(4, 8, dtype=torch.long)
    targets = torch.tensor([0, 1, 2, 1], dtype=torch.long)

    collector = KFACFactorCollector(model)
    stats = collector.accumulate_batch(
        criterion,
        (input_ids, attention_mask),
        targets,
    )

    for name in (
        "layers.0.q_proj",
        "layers.0.k_proj",
        "layers.0.v_proj",
        "layers.0.out_proj",
    ):
        assert name in stats
        assert stats[name].info.role == "attention_projection"
        assert stats[name].info.num_heads == 4
        assert stats[name].activation_cov.shape == (17, 17)
        assert stats[name].gradient_cov.shape == (16, 16)


def test_kfac_collector_accumulates_over_loader():
    model = nn.Sequential(
        nn.Linear(4, 5),
        nn.ReLU(),
        nn.Linear(5, 2),
    ).double()
    criterion = nn.CrossEntropyLoss()
    inputs = torch.randn(6, 4, dtype=torch.float64)
    targets = torch.tensor([0, 1, 0, 1, 1, 0], dtype=torch.long)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=3, shuffle=False)

    collector = KFACFactorCollector(model)
    stats = collector.accumulate_loader(loader, criterion)

    assert "0" in stats
    assert "2" in stats
    assert stats["0"].num_samples > 0
    assert torch.isfinite(stats["0"].activation_cov).all()
    assert torch.isfinite(stats["0"].gradient_cov).all()
