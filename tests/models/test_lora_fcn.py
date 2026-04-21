import torch

from gif.models import (
    FullyConnectedNet,
    LoRAFullyConnectedNet,
    count_trainable_parameters,
    load_base_state_dict_into_lora,
)


def test_lora_fcn_matches_base_output_at_zero_adapter():
    torch.manual_seed(0)
    base = FullyConnectedNet(4, 8, 3, 3, 0.0).double()
    lora = LoRAFullyConnectedNet(4, 8, 3, 3, 0.0, lora_rank=2, lora_alpha=4.0).double()
    load_base_state_dict_into_lora(lora, base.state_dict())

    inputs = torch.randn(5, 4, dtype=torch.float64)
    with torch.no_grad():
        base_outputs = base(inputs)
        lora_outputs = lora(inputs)

    assert torch.allclose(base_outputs, lora_outputs, atol=1e-10, rtol=1e-10)


def test_lora_fcn_has_fewer_trainable_parameters_than_full_model():
    base = FullyConnectedNet(4, 8, 3, 3, 0.0)
    lora = LoRAFullyConnectedNet(4, 8, 3, 3, 0.0, lora_rank=2, lora_alpha=4.0)

    full_trainable = sum(parameter.numel() for parameter in base.parameters())
    lora_trainable = count_trainable_parameters(lora)

    assert lora_trainable < full_trainable
    assert lora_trainable > 0
