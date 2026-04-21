from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .fcn import FullyConnectedNet


class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        bias: bool = True,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive.")

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.weight.requires_grad_(False)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            self.bias.requires_grad_(False)
        else:
            self.register_parameter("bias", None)

        self.lora_down = nn.Parameter(torch.empty(rank, in_features))
        self.lora_up = nn.Parameter(torch.empty(out_features, rank))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        nn.init.normal_(self.lora_down, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_up)

    @classmethod
    def from_linear(cls, layer: nn.Linear, rank: int, alpha: float) -> "LoRALinear":
        lora = cls(
            in_features=layer.in_features,
            out_features=layer.out_features,
            rank=rank,
            alpha=alpha,
            bias=layer.bias is not None,
        )
        with torch.no_grad():
            lora.weight.copy_(layer.weight)
            if layer.bias is not None:
                lora.bias.copy_(layer.bias)
        return lora

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        adapter = F.linear(F.linear(x, self.lora_down), self.lora_up)
        return base + self.scaling * adapter


class LoRAFullyConnectedNet(FullyConnectedNet):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        dropout_prob: float,
        lora_rank: int,
        lora_alpha: float,
    ):
        super().__init__(input_size, hidden_size, output_size, num_layers, dropout_prob)
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        for index, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                self.layers[index] = LoRALinear.from_linear(
                    layer, rank=lora_rank, alpha=lora_alpha
                )


def get_trainable_parameters(module: nn.Module) -> list[nn.Parameter]:
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in get_trainable_parameters(module))


def trainable_parameters_to_vector(module: nn.Module) -> torch.Tensor:
    params = get_trainable_parameters(module)
    if not params:
        raise RuntimeError("No trainable parameters were found.")
    return parameters_to_vector(params)


def vector_to_trainable_parameters(vector: torch.Tensor, module: nn.Module) -> None:
    params = get_trainable_parameters(module)
    if not params:
        raise RuntimeError("No trainable parameters were found.")
    vector_to_parameters(vector, params)


def load_base_state_dict_into_lora(module: nn.Module, base_state_dict: dict[str, torch.Tensor]) -> None:
    missing_keys = []
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if name.endswith(".weight") or name.endswith(".bias"):
                base_value = base_state_dict.get(name)
                if base_value is None:
                    missing_keys.append(name)
                    continue
                parameter.copy_(base_value)
    if missing_keys:
        raise KeyError(f"Missing base parameters for LoRA initialization: {missing_keys}")


__all__ = [
    "LoRALinear",
    "LoRAFullyConnectedNet",
    "count_trainable_parameters",
    "get_trainable_parameters",
    "load_base_state_dict_into_lora",
    "trainable_parameters_to_vector",
    "vector_to_trainable_parameters",
]
