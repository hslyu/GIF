import numpy as np
import torch

from gif.influence import HyperInfluence, hyperinf_update
from gif.models import FullyConnectedNet
from gif.selection import HighestKGradients


def _build_toy_model():
    model = FullyConnectedNet(
        input_size=4,
        hidden_size=8,
        output_size=3,
        num_layers=3,
        dropout_prob=0.0,
    ).double()
    criterion = torch.nn.CrossEntropyLoss()
    return model, criterion


def test_hyperinf_update_returns_projected_subset_shape():
    torch.manual_seed(0)
    model, criterion = _build_toy_model()
    inputs = torch.randn(6, 4, dtype=torch.float64)
    targets = torch.tensor([0, 1, 2, 1, 0, 2], dtype=torch.long)

    selector = HighestKGradients(model, ratio=0.2)
    selector.register_hooks()
    try:
        loss = criterion(model(inputs), targets)
        loss.backward()
        index_list = selector.get_parameters()
    finally:
        selector.remove_hooks()

    update = hyperinf_update(
        model=model,
        total_loss=criterion(model(inputs), targets),
        target_loss=criterion(model(inputs[:2]), targets[:2]),
        index_list=np.array(index_list, dtype=int),
        max_iter=4,
        tol=1e-8,
    )

    assert update.ndim == 1
    assert update.shape[0] == len(index_list)
    assert torch.isfinite(update).all()


def test_hyperinf_update_moves_target_loss_in_expected_direction():
    torch.manual_seed(1)
    model, criterion = _build_toy_model()
    inputs = torch.randn(8, 4, dtype=torch.float64)
    targets = torch.tensor([0, 1, 2, 1, 0, 2, 1, 0], dtype=torch.long)

    selector = HighestKGradients(model, ratio=1.0)
    selector.register_hooks()
    try:
        total_loss = criterion(model(inputs), targets)
        total_loss.backward()
        index_list = np.array(selector.get_parameters(), dtype=int)
    finally:
        selector.remove_hooks()

    total_loss = criterion(model(inputs), targets)
    target_loss = criterion(model(inputs[:3]), targets[:3])
    update = hyperinf_update(
        model=model,
        total_loss=total_loss,
        target_loss=target_loss,
        index_list=index_list,
        max_iter=4,
        tol=1e-8,
    )

    before = target_loss.item()
    selector = HighestKGradients(model, ratio=1.0)
    selector.register_hooks()
    try:
        loss = criterion(model(inputs), targets)
        loss.backward()
        selector.get_parameters()
        selector.update_network(1e-4 * update)
    finally:
        selector.remove_hooks()

    after = criterion(model(inputs[:3]), targets[:3]).item()
    assert after > before


def test_hyperinf_class_api_returns_details_and_keeps_model_state():
    torch.manual_seed(2)
    model, criterion = _build_toy_model()
    inputs = torch.randn(6, 4, dtype=torch.float64)
    targets = torch.tensor([0, 1, 2, 1, 0, 2], dtype=torch.long)
    original = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    selector = HighestKGradients(model, ratio=0.3)
    selector.register_hooks()
    try:
        loss = criterion(model(inputs), targets)
        loss.backward()
        index_list = np.array(selector.get_parameters(), dtype=int)
    finally:
        selector.remove_hooks()

    result = HyperInfluence().compute(
        model=model,
        total_loss=criterion(model(inputs), targets),
        target_loss=criterion(model(inputs[:2]), targets[:2]),
        index_list=index_list,
        max_iter=4,
        tol=1e-8,
        return_details=True,
    )

    assert "update" in result
    assert "details" in result
    assert torch.isfinite(result["update"]).all()
    for name, tensor in original.items():
        assert torch.equal(model.state_dict()[name], tensor)
