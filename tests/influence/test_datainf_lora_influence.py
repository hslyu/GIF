import torch

from gif.influence import DataInfluence, datainf_update
from gif.models import (
    FullyConnectedNet,
    LoRAFullyConnectedNet,
    load_base_state_dict_into_lora,
    trainable_parameters_to_vector,
    vector_to_trainable_parameters,
)


def _build_lora_model():
    torch.manual_seed(0)
    base = FullyConnectedNet(4, 8, 3, 3, 0.0).double()
    lora = LoRAFullyConnectedNet(4, 8, 3, 3, 0.0, lora_rank=2, lora_alpha=4.0).double()
    load_base_state_dict_into_lora(lora, base.state_dict())
    return lora, torch.nn.CrossEntropyLoss()


def test_datainf_update_matches_trainable_parameter_shape():
    model, criterion = _build_lora_model()
    inputs = torch.randn(6, 4, dtype=torch.float64)
    targets = torch.tensor([0, 1, 2, 1, 0, 2], dtype=torch.long)

    update = datainf_update(
        model=model,
        total_loss=criterion(model(inputs), targets),
        target_loss=criterion(model(inputs[:2]), targets[:2]),
        damping=1e-6,
    )

    assert update.shape == trainable_parameters_to_vector(model).shape
    assert torch.isfinite(update).all()


def test_datainf_update_moves_target_loss_upward():
    model, criterion = _build_lora_model()
    inputs = torch.randn(8, 4, dtype=torch.float64)
    targets = torch.tensor([0, 1, 2, 1, 0, 2, 1, 0], dtype=torch.long)

    total_loss = criterion(model(inputs), targets)
    target_loss = criterion(model(inputs[:3]), targets[:3])
    update = datainf_update(
        model=model,
        total_loss=total_loss,
        target_loss=target_loss,
        damping=1e-6,
    )

    before = target_loss.item()
    base_vector = trainable_parameters_to_vector(model).detach()
    vector_to_trainable_parameters(base_vector + 1e-4 * update, model)
    after = criterion(model(inputs[:3]), targets[:3]).item()

    assert after > before


def test_datainf_class_api_returns_details():
    model, criterion = _build_lora_model()
    inputs = torch.randn(6, 4, dtype=torch.float64)
    targets = torch.tensor([0, 1, 2, 1, 0, 2], dtype=torch.long)

    result = DataInfluence().compute(
        model=model,
        total_loss=criterion(model(inputs), targets),
        target_loss=criterion(model(inputs[:2]), targets[:2]),
        damping=1e-6,
        return_details=True,
    )

    assert "update" in result
    assert "details" in result
    assert torch.isfinite(result["update"]).all()


def test_datainf_update_supports_full_model_training():
    torch.manual_seed(1)
    model = FullyConnectedNet(4, 8, 3, 3, 0.0).double()
    criterion = torch.nn.CrossEntropyLoss()
    inputs = torch.randn(8, 4, dtype=torch.float64)
    targets = torch.tensor([0, 1, 2, 1, 0, 2, 1, 0], dtype=torch.long)

    update = datainf_update(
        model=model,
        total_loss=criterion(model(inputs), targets),
        target_loss=criterion(model(inputs[:3]), targets[:3]),
        damping=1e-6,
    )

    expected_dim = sum(parameter.numel() for parameter in model.parameters())
    assert update.ndim == 1
    assert update.shape[0] == expected_dim
    assert torch.isfinite(update).all()
