import numpy as np
import torch
import torch.nn as nn

from .abstract_selection import Selection, _ModuleInfo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Random(Selection):
    def __init__(self, net: torch.nn.Module, ratio):
        assert 0 < ratio <= 1, "ratio should be in (0, 1]"
        super(Random, self).__init__()
        self.net = net
        self.ratio = ratio
        self.module_info_list = self._get_module_info_list()

    def _get_module_info_list(self):
        module_info_list = []
        start_index = 0
        for module in self.net.modules():
            if not self._is_single_layer(module):
                continue

            module_size = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                num_weight_params = int(module.weight.numel() * self.ratio)
                weight_index_list = np.random.choice(
                    np.arange(module.weight.numel()), num_weight_params, replace=False
                )

                num_bias_params = 0
                bias_index_list = np.empty(0, dtype=int)
                if module.bias is not None:
                    num_bias_params = int(module.bias.numel() * self.ratio)
                    bias_index_list = np.random.choice(
                        np.arange(module.bias.numel()), num_bias_params, replace=False
                    )
                index_list = np.concatenate(
                    (weight_index_list, bias_index_list + module.weight.numel())
                )
                num_params = num_weight_params + num_bias_params
                module_info = _ModuleInfo(
                    module=module,
                    start_index=start_index,
                    num_params=num_params,
                    index_list=index_list,
                    weight_index_list=weight_index_list,
                    bias_index_list=bias_index_list,
                )

                module_info_list.append(module_info)

            start_index += module_size

        return module_info_list

    def _is_single_layer(self, module):
        return list(module.children()) == []

    def get_parameters(self):
        selected_parameter_indices = np.empty(0, dtype=int)
        for info in self.module_info_list:
            selected_parameter_indices = np.concatenate(
                (selected_parameter_indices, info.index_list + info.start_index)
            )

        return selected_parameter_indices

    def update_network(self, vectorized_influence):
        assert sum(info.num_params for info in self.module_info_list) == len(
            vectorized_influence
        ), f"length of vectorized_influence {len(vectorized_influence)} is not equal to the number of seleceted parameters {sum(info.num_params for info in self.module_info_list)}"

        with torch.no_grad():
            current = 0
            for info in self.module_info_list:
                module = info.module
                change_list = vectorized_influence[current : current + info.num_params]
                current += info.num_params

                weight_change = torch.zeros(module.weight.numel()).to(device)
                weight_change[info.weight_index_list] = change_list[
                    : len(info.weight_index_list)
                ]
                module.weight.data += weight_change.view_as(module.weight.data)

                if module.bias is not None:
                    bias_change = torch.zeros(module.bias.numel()).to(device)
                    bias_change[info.bias_index_list] = change_list[
                        len(info.weight_index_list) :
                    ]
                    module.bias.data += bias_change.view_as(module.bias.data)
