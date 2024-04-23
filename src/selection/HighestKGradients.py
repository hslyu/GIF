import numpy as np
import torch
from torch import nn

from .abstract_selection import Selection, _ModuleInfo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HighestKGradients(Selection):
    def __init__(self, net, ratio):
        assert 0 < ratio <= 1, "ratio should be in (0, 1]"
        super(HighestKGradients, self).__init__()
        self.net = net
        self.ratio = ratio
        self.hook_handle_list = []
        self.module_info_list = []

    def generate_hook(self, start_index):
        def hook(module, grad_input, grad_output):
            module_size = sum(p.numel() for p in module.parameters() if p.requires_grad)
            num_params = int(module_size * self.ratio)
            module_info = _ModuleInfo(module, start_index, num_params)
            selected_index_list = np.empty(0, dtype=int)

            if isinstance(module, nn.Linear):
                # Compute the allotted number of neurons for each rows
                num_weights_per_output = module.weight.size(1)
                # Get list of indices of neurons with highest activation
                batch_abs_mean = torch.abs(torch.mean(grad_output[0], 0))
            else:  # isinstance(module, nn.Conv2d):
                num_weights_per_output = (
                    module.weight.size(1)
                    * module.weight.size(2)
                    * module.weight.size(3)
                )
                # Get list of indices of neurons with highest activation
                batch_abs_mean = torch.abs(torch.mean(grad_output[0], (0, 2, 3)))

            if module.bias is not None:
                num_required_indices = num_params // (num_weights_per_output + 1)
                leftover = num_params % (num_weights_per_output + 1)
            else:
                num_required_indices = num_params // (num_weights_per_output)
                leftover = num_params % (num_weights_per_output)

            index_list = torch.sort(batch_abs_mean, descending=True)[1]
            # Add the indices of weights
            for index in index_list[:num_required_indices]:
                selected_index_list = np.concatenate(
                    (
                        selected_index_list,
                        np.arange(num_weights_per_output)
                        + num_weights_per_output * index.item(),
                    )
                )

            # Add the indices of weights for the leftover neurons
            if leftover != 0:
                index = index_list[num_required_indices]
                # random pick leftover number of neurons
                indices = (
                    np.random.choice(
                        np.arange(num_weights_per_output), leftover, replace=False
                    )
                    + num_weights_per_output * index.item()
                )
                selected_index_list = np.concatenate((selected_index_list, indices))

            module_info.weight_index_list = selected_index_list

            if module.bias is not None:
                module_info.bias_index_list = (
                    index_list[:num_required_indices].detach().cpu().numpy()
                )
                # Add the indices of the bias
                selected_index_list = np.concatenate(
                    (
                        selected_index_list,
                        module_info.bias_index_list + module.weight.numel(),
                    )
                )
            module_info.index_list = selected_index_list
            self.module_info_list.append(module_info)

        return hook

    def register_hooks(self):
        start_index = 0
        for module in self.net.modules():
            if not self._is_single_layer(module):
                continue

            num_param = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                hook_fn = self.generate_hook(start_index)
                hook_handle = module.register_full_backward_hook(hook_fn)
                self.hook_handle_list.append(hook_handle)

            start_index += num_param
        return self.hook_handle_list

    def remove_hooks(self):
        for handle in self.hook_handle_list:
            handle.remove()

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
