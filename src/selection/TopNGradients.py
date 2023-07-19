import numpy as np
import torch
from torch import nn

from .abstract_selection import Selection


class TopNGradients(Selection):
    def __init__(self, net, num_choices):
        self.hooks = []
        self.num_choices = num_choices
        self.chosen_param_list = np.zeros(num_choices, dtype="int32")
        self.current = 0
        self.require_backward = True

        self.sum = 0
        self.net = net
        self.num_params = self._compute_num_param()
        for module in net.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                self.first_module = module
                break

    def generate_hook(self, start_index):
        def hook(module, grad_input, grad_output):
            # If statement stops to call hook function twice when I use retain_graph=True.
            if self.current >= self.num_choices:
                return
            module_size = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if module is self.first_module:
                quota = int(self.num_choices - self.current)
            else:
                quota = int(module_size * self.num_choices / self.num_params)

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
                num_required_indices = quota // (num_weights_per_output + 1)
                leftover = quota % (num_weights_per_output + 1)
            else:
                num_required_indices = quota // (num_weights_per_output)
                leftover = quota % (num_weights_per_output)
            index_list = torch.sort(batch_abs_mean, descending=True)[1]
            # Add the indices of weights
            for index in index_list[:num_required_indices]:
                self.chosen_param_list[
                    self.current : self.current + num_weights_per_output
                ] = (
                    np.arange(num_weights_per_output)
                    + start_index
                    + num_weights_per_output * index.item()
                )
                self.current += num_weights_per_output

            # Add the indices of weights for the leftover neurons
            if leftover != 0:
                index = index_list[num_required_indices]
                weight = module.weight[index].flatten()
                _, indices = torch.sort(torch.abs(weight), descending=True)
                # Add the start position of the module when network is flattened.
                indices = (
                    indices.detach().cpu().numpy()
                    + start_index  # start of the module
                    + num_weights_per_output * index.item()
                )
                self.chosen_param_list[
                    self.current : self.current + leftover
                ] = indices[:leftover]
                self.current += leftover

            if module.bias is not None:
                # Add the indices of the bias
                self.chosen_param_list[
                    self.current : self.current + num_required_indices
                ] = (
                    index_list[:num_required_indices].detach().cpu().numpy()
                    + start_index
                    + len(module.weight.flatten())
                )
                self.current += num_required_indices

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
                self.hooks.append(hook_handle)

            start_index += num_param
        return self.hooks

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def initialize_neurons(self):
        self.chosen_param_list = np.zeros(self.num_choices, dtype="int32")
        self.current = 0

    def _compute_num_param(self):
        num_param = 0
        for module in self.net.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                num_param += sum(
                    p.numel() for p in module.parameters() if p.requires_grad
                )
        return num_param

    def _is_single_layer(self, module):
        return list(module.children()) == []

    def get_parameters(self):
        return self.chosen_param_list
