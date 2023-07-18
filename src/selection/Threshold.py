import numpy as np
import torch
from torch import nn

from .abstract_selection import Selection


class Threshold(Selection):
    def __init__(self, net, num_choices, threshold=1.0):
        # super(Threshold, self).__init__()
        self.hooks = []
        self.num_choices = num_choices
        self.chosen_param_list = np.zeros(num_choices, dtype="int16")
        self.current = 0
        self.threshold = threshold

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

            index_list = torch.empty(0)
            num_elements = 0
            while num_elements < quota:
                index_list = torch.where(batch_abs_mean > self.threshold)[0]
                num_elements = len(index_list) * (num_weights_per_output + 1)
                self.threshold -= 0.01

            param_pool = np.empty(0)
            # weights
            for index in index_list:
                param_indices = (
                    np.arange(num_weights_per_output)
                    + start_index
                    + index.item() * num_weights_per_output
                )
                param_pool = np.append(param_pool, param_indices)

            # bias
            param_pool = np.append(
                param_pool,
                index_list.detach().cpu().numpy()
                + len(module.weight.flatten())
                + start_index,
            )

            self.chosen_param_list[
                self.current : self.current + quota
            ] = np.random.choice(param_pool, quota, replace=False)
            self.current += quota

        return hook

    def register_hooks(self):
        start_index = 0
        for module in self.net.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                hook_fn = self.generate_hook(start_index)
                hook_handle = module.register_forward_hook(hook_fn)
                start_index += sum(p.numel() for p in module.parameters())
                self.hooks.append(hook_handle)
        return self.hooks

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def initialize_neurons(self):
        self.chosen_param_list = np.zeros(self.num_choices, dtype="int16")
        self.current = 0

    def get_parameters(self):
        return self.chosen_param_list