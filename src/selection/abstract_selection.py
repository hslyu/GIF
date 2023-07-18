from abc import ABCMeta

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
