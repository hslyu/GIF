import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from gif.selection import CAPS, HighestKGradients


class TinySelectionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=3, bias=True)
        self.fc = nn.Linear(8, 3, bias=True)

    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class TinySelectionNetWithBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=3, bias=True)
        self.bn = nn.BatchNorm2d(2)
        self.fc = nn.Linear(8, 3, bias=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = torch.relu(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def _num_trainable_params(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _num_selectable_params(model):
    return sum(
        parameter.numel()
        for module in model.modules()
        if list(module.children()) == [] and isinstance(module, (nn.Conv2d, nn.Linear))
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def _flatten_trainable_params(model):
    return torch.cat(
        [parameter.detach().flatten().clone() for parameter in model.parameters() if parameter.requires_grad]
    )


def _assert_full_selection(index_list, expected_count):
    assert len(index_list) == expected_count
    assert len(np.unique(index_list)) == expected_count
    assert np.array_equal(np.sort(index_list), np.arange(expected_count))


def test_highest_k_gradients_selects_all_params_when_ratio_is_one():
    model = TinySelectionNet()
    selector = HighestKGradients(model, ratio=1.0)
    criterion = nn.CrossEntropyLoss()

    selector.register_hooks()
    inputs = torch.randn(4, 1, 4, 4)
    targets = torch.tensor([0, 1, 2, 1])
    loss = criterion(model(inputs), targets)
    loss.backward()

    try:
        index_list = selector.get_parameters()
        _assert_full_selection(index_list, _num_trainable_params(model))
    finally:
        selector.remove_hooks()


def test_highest_k_gradients_updates_all_params_when_ratio_is_one():
    model = TinySelectionNet()
    selector = HighestKGradients(model, ratio=1.0)
    criterion = nn.CrossEntropyLoss()

    selector.register_hooks()
    inputs = torch.randn(4, 1, 4, 4)
    targets = torch.tensor([0, 1, 2, 1])
    loss = criterion(model(inputs), targets)
    loss.backward()

    before = _flatten_trainable_params(model)
    try:
        index_list = selector.get_parameters()
        update = torch.full((len(index_list),), 0.01)
        selector.update_network(update)
    finally:
        selector.remove_hooks()

    after = _flatten_trainable_params(model)
    assert torch.allclose(after - before, torch.full_like(before, 0.01))


def test_caps_selects_all_params_when_ratio_is_one():
    model = TinySelectionNet()
    selector = CAPS(model, ratio=1.0, min_curv=0.0)
    criterion = nn.CrossEntropyLoss()

    inputs = torch.randn(4, 1, 4, 4)
    targets = torch.tensor([0, 1, 2, 1])
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)

    selector.fit(loader, loader, criterion)
    index_list = selector.get_parameters()

    _assert_full_selection(index_list, _num_trainable_params(model))


def test_caps_updates_all_params_when_ratio_is_one():
    model = TinySelectionNet()
    selector = CAPS(model, ratio=1.0, min_curv=0.0)
    criterion = nn.CrossEntropyLoss()

    inputs = torch.randn(4, 1, 4, 4)
    targets = torch.tensor([0, 1, 2, 1])
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)

    selector.fit(loader, loader, criterion)
    index_list = selector.get_parameters()

    before = _flatten_trainable_params(model)
    update = torch.full((len(index_list),), 0.01)
    selector.update_network(update)
    after = _flatten_trainable_params(model)

    assert torch.allclose(after - before, torch.full_like(before, 0.01))


def test_highest_k_gradients_ratio_one_selects_all_supported_params_but_not_batchnorm():
    model = TinySelectionNetWithBatchNorm()
    selector = HighestKGradients(model, ratio=1.0)
    criterion = nn.CrossEntropyLoss()

    selector.register_hooks()
    inputs = torch.randn(4, 1, 4, 4)
    targets = torch.tensor([0, 1, 2, 1])
    loss = criterion(model(inputs), targets)
    loss.backward()

    try:
        index_list = selector.get_parameters()
    finally:
        selector.remove_hooks()

    assert len(index_list) == _num_selectable_params(model)
    assert len(index_list) < _num_trainable_params(model)


def test_caps_ratio_one_selects_all_supported_params_but_not_batchnorm():
    model = TinySelectionNetWithBatchNorm()
    selector = CAPS(model, ratio=1.0, min_curv=0.0)
    criterion = nn.CrossEntropyLoss()

    inputs = torch.randn(4, 1, 4, 4)
    targets = torch.tensor([0, 1, 2, 1])
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)

    selector.fit(loader, loader, criterion)
    index_list = selector.get_parameters()

    assert len(index_list) == _num_selectable_params(model)
    assert len(index_list) < _num_trainable_params(model)
