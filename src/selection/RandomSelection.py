import numpy as np
import torch
import torch.nn as nn

from .abstract_selection import Selection


class RandomSelection(Selection):
    def __init__(self, net: torch.nn.Module, num_choices):
        self.net = net
        self.num_choices = num_choices

    def get_parameters(self):
        param_index = self._get_index_list()
        return np.random.choice(param_index, self.num_choices, replace=False).astype(
            int
        )

    def _get_index_list(self):
        index_list = np.array([])
        start_index = 0
        for module in self.net.modules():
            if not self._is_single_layer(module):
                continue

            num_param = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                module_index_list = np.arange(num_param, dtype=int) + start_index
                index_list = np.append(index_list, module_index_list)

            start_index += num_param

        return index_list

    def _is_single_layer(self, module):
        return list(module.children()) == []
