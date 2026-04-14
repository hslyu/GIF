import numpy as np
import torch
from torch import nn

from .abstract_selection import Selection, _ModuleInfo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Threshold(Selection):
    def __init__(self, net, ratio, threshold=1.0):
        super(Threshold, self).__init__()
        self.net = net
        self.ratio = ratio
        self.threshold = threshold
        self.hook_handle_list = []
        self.module_info_list = []

    def generate_hook(self, start_index):
        def hook(module, input, output):
            # Determine the number of neurons to be chosen
            num_weight_params = int(self.ratio * module.weight.numel())
            num_bias_params = (
                0 if module.bias is None else int(self.ratio * module.bias.numel())
            )
            num_params = num_weight_params + num_bias_params

            if isinstance(module, nn.Linear):
                # Compute the allotted number of neurons for each rows
                num_weights_per_output = module.weight.size(1)
                # Get list of indices of neurons with highest activation
                batch_abs_mean = torch.abs(torch.mean(output, 0))
            else:  # isinstance(module, nn.Conv2d):
                num_weights_per_output = (
                    module.weight.size(1)
                    * module.weight.size(2)
                    * module.weight.size(3)
                )
                # Get list of indices of neurons with highest activation
                batch_abs_mean = torch.abs(torch.mean(output, (0, 2, 3)))

            threshold_index_list = torch.empty(0)
            num_elements = 0
            while num_elements < num_params:
                threshold_index_list = torch.where(batch_abs_mean > self.threshold)[0]
                if module.bias is None:
                    num_elements = len(threshold_index_list) * num_weights_per_output
                else:
                    num_elements = len(threshold_index_list) * (
                        num_weights_per_output + 1
                    )
                self.threshold -= 0.01

            weight_index_pool = np.empty(0)
            for index in threshold_index_list:
                weight_index_pool = np.concatenate(
                    (
                        weight_index_pool,
                        np.arange(num_weights_per_output)
                        + num_weights_per_output * index.item(),
                    )
                )

            weight_index_list = np.random.choice(
                weight_index_pool, num_weight_params, replace=False
            )

            bias_index_list = np.empty(0, dtype=int)
            if module.bias is not None:
                bias_index_list = np.random.choice(
                    threshold_index_list.detach().cpu().numpy(),
                    num_bias_params,
                    replace=False,
                )

            index_list = np.concatenate(
                (weight_index_list, bias_index_list + module.weight.numel())
            )
            module_info = _ModuleInfo(
                module=module,
                start_index=start_index,
                num_params=num_params,
                index_list=index_list,
                weight_index_list=weight_index_list,
                bias_index_list=bias_index_list,
            )
            self.module_info_list.append(module_info)

        return hook

    def register_hooks(self):
        start_index = 0
        for module in self.net.modules():
            if not self._is_single_layer(module):
                continue

            module_size = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                hook_fn = self.generate_hook(start_index)
                hook_handle = module.register_forward_hook(hook_fn)
                self.hook_handle_list.append(hook_handle)

            start_index += module_size
        return self.hook_handle_list

    def remove_hooks(self):
        for hook in self.hook_handle_list:
            hook.remove()

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
