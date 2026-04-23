import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from gif.models import TextTransformerClassifier
from gif.selection import EKFACCAPS


class _TextBatchDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (self.input_ids[index], self.attention_mask[index]), self.labels[index]


def test_ekfac_caps_selects_blocks_for_fcn():
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(4, 6),
        nn.ReLU(),
        nn.Linear(6, 3),
    ).double()
    criterion = nn.CrossEntropyLoss()

    target_inputs = torch.randn(8, 4, dtype=torch.float64)
    target_targets = torch.tensor([0, 1, 2, 1, 0, 2, 1, 0], dtype=torch.long)
    retained_inputs = torch.randn(10, 4, dtype=torch.float64)
    retained_targets = torch.tensor([0, 1, 2, 1, 0, 2, 1, 0, 2, 1], dtype=torch.long)

    target_loader = DataLoader(
        TensorDataset(target_inputs, target_targets),
        batch_size=4,
        shuffle=False,
    )
    retained_loader = DataLoader(
        TensorDataset(retained_inputs, retained_targets),
        batch_size=5,
        shuffle=False,
    )

    selector = EKFACCAPS(model, ratio=0.5, lam=1e-3)
    selector.fit(
        target_loader=target_loader,
        retained_loader=retained_loader,
        criterion=criterion,
        device=torch.device("cpu"),
    )

    selected = selector.get_parameters()
    assert len(selected) > 0
    assert 0.0 <= selector.gradient_energy_coverage <= 1.0
    assert selector.total_gradient_energy >= selector.selected_gradient_energy >= 0.0


def test_ekfac_caps_supports_attention_projection_blocks():
    torch.manual_seed(0)
    model = TextTransformerClassifier(
        vocab_size=64,
        max_len=8,
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        num_classes=3,
        dropout_prob=0.0,
    ).double()
    criterion = nn.CrossEntropyLoss()

    target_ids = torch.randint(1, 64, (6, 8), dtype=torch.long)
    target_mask = torch.ones(6, 8, dtype=torch.long)
    target_targets = torch.tensor([0, 1, 2, 1, 0, 2], dtype=torch.long)
    retained_ids = torch.randint(1, 64, (8, 8), dtype=torch.long)
    retained_mask = torch.ones(8, 8, dtype=torch.long)
    retained_targets = torch.tensor([0, 1, 2, 1, 0, 2, 1, 0], dtype=torch.long)

    target_loader = DataLoader(
        _TextBatchDataset(target_ids, target_mask, target_targets),
        batch_size=3,
        shuffle=False,
    )
    retained_loader = DataLoader(
        _TextBatchDataset(retained_ids, retained_mask, retained_targets),
        batch_size=4,
        shuffle=False,
    )

    selector = EKFACCAPS(model, ratio=0.2, lam=1e-3)
    selector.fit(
        target_loader=target_loader,
        retained_loader=retained_loader,
        criterion=criterion,
        device=torch.device("cpu"),
    )

    selected = selector.get_parameters()
    assert len(selected) > 0
    assert selector.gradient_energy_coverage > 0.0
