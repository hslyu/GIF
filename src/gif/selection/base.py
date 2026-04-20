from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import torch


class Selection(metaclass=ABCMeta):
    net: torch.nn.Module
    num_choices: int
    require_backward: bool = False

    @abstractmethod
    def get_parameters(self):
        raise NotImplementedError

    def register_hooks(self):
        return []

    def remove_hooks(self):
        return None

    def initialize_neurons(self):
        return None

    @abstractmethod
    def update_network(self, vectorized_influence):
        raise NotImplementedError


@dataclass
class _ModuleInfo:
    module: torch.nn.Module
    start_index: int
    num_params: int
    index_list: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    weight_index_list: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=int)
    )
    bias_index_list: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
