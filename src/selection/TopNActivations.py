import numpy as np
import torch
from torch import nn

from .abstract_selection import Selection


class TopNActivations(Selection):
    def __init__(self, net, num_choices):
        super(TopNActivations, self).__init__()
        self.hooks = []
        self.num_choices = num_choices
        self.chosen_param_list = np.zeros(num_choices, dtype="int32")
        self.current = 0

        self.net = net
        self.num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        self.last_module = list(net.modules())[-1]

    def generate_hook(self, start_index):
        def hook(module, input, output):
            # Determine the number of neurons to be chosen
            module_size = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if module is self.last_module:
                quota = int(self.num_choices - self.current)
            else:
                quota = int(module_size * self.num_choices / self.num_params)

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

            num_required_indices = quota // (num_weights_per_output + 1)
            leftover = quota % (num_weights_per_output + 1)
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
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                hook_fn = self.generate_hook(start_index)
                hook_handle = module.register_forward_hook(hook_fn)
                start_index += sum(
                    p.numel() for p in module.parameters() if p.requires_grad
                )
                self.hooks.append(hook_handle)
            elif (
                isinstance(module, nn.BatchNorm1d)
                or isinstance(module, nn.BatchNorm2d)
                or isinstance(module, nn.BatchNorm3d)
            ):
                start_index += sum(
                    p.numel() for p in module.parameters() if p.requires_grad
                )
        return self.hooks

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def initialize_neurons(self):
        self.chosen_param_list = np.zeros(self.num_choices, dtype="int32")
        self.current = 0

    def get_parameters(self):
        return self.chosen_param_list
