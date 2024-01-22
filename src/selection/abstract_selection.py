from abc import ABCMeta
from dataclasses import dataclass

import numpy as np
import torch


class Selection(metaclass=ABCMeta):
    net: torch.nn.Module
    num_choices: int
    require_backward: bool = False

    def get_parameters(self):
        pass

    def register_hooks(self):
        pass

    def remove_hooks(self):
        pass

    def initialize_neurons(self):
        pass


@dataclass
class _ModuleInfo:
    module: torch.nn.Module
    start_index: int
    num_params: int
    index_list: np.ndarray = np.empty(0, dtype=int)
    weight_index_list: np.ndarray = np.empty(0, dtype=int)
    bias_index_list: np.ndarray = np.empty(0, dtype=int)
