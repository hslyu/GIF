import numpy as np
import torch

from .abstract_selection import Selection


class RandomSelection(Selection):
    def __init__(self, net: torch.nn.Module, num_choices):
        self.net = net
        self.num_choices = num_choices

    def get_parameters(self):
        num_param = sum(p.numel() for p in self.net.parameters())
        return np.random.choice(num_param, self.num_choices, replace=False)
