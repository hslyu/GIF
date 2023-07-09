import numpy as np
import torch
from torch import nn


class TopNActivations(nn.Module):
    def __init__(self, net, num_choices):
        super(TopNActivations, self).__init__()
        self.hooks = []
        self.num_choices = num_choices
        self.chosen_param_list = np.zeros(num_choices, dtype="int16")
        self.current = 0

        self.net = net
        self.num_params = sum(p.numel() for p in net.parameters())
        self.last_module = list(net.modules())[-1]

    def activation_hook(self, start_index):
        def hook(module, input, output):
            # Determine the number of neurons to be chosen
            module_size = len(module.weight.view(-1))
            if module is self.last_module:
                quota = self.num_choices - self.current
            else:
                quota = int(self.num_choices * module_size / self.num_params)

            if isinstance(module, nn.Linear):
                # Compute the allotted number of neurons for each rows
                num_weights_per_index = module.weight.size(1)
                required_rows = quota // num_weights_per_index + 1

                # Get list of indices of neurons with highest activation
                batch_mean = torch.mean(output, 0)
                index_list = torch.sort(batch_mean, descending=True)[1][:required_rows]

                # Get the indices of the weights corresponding to the neurons
                for index in index_list:
                    weight = module.weight[index]
                    _, indices = torch.sort(torch.abs(weight), descending=True)
                    # Add the start position of the module when network is flattened.
                    indices = (
                        indices.detach().cpu().numpy()
                        + start_index  # start of the module
                        + num_weights_per_index * index.item()  # start of the index
                    )

                    num_saving = (
                        quota - num_weights_per_index * (required_rows - 1)
                        if index == index_list[-1]
                        else num_weights_per_index
                    )
                    self.chosen_param_list[
                        self.current : self.current + num_saving
                    ] = indices[:num_saving]
                    self.current += num_saving

            elif isinstance(module, nn.Conv2d):
                num_weights_per_kernel = (
                    module.weight.size(1)
                    * module.weight.size(2)
                    * module.weight.size(3)
                )
                required_kernels = quota // num_weights_per_kernel + 1

                batch_mean = torch.mean(output, (0, 2, 3))
                index_list = torch.sort(batch_mean, descending=True)[1][
                    :required_kernels
                ]

                # Get the indices of the weights corresponding to the neurons
                for index in index_list:
                    weight = module.weight[index].flatten()
                    _, indices = torch.sort(torch.abs(weight), descending=True)
                    # Add the start position of the module when network is flattened.
                    indices = (
                        indices.detach().cpu().numpy()
                        + start_index  # start of the module
                        + num_weights_per_kernel * index.item()  # start of the index
                    )

                    num_saving = (
                        quota - num_weights_per_kernel * (required_kernels - 1)
                        if index == index_list[-1]
                        else num_weights_per_kernel
                    )
                    self.chosen_param_list[
                        self.current : self.current + num_saving
                    ] = indices[:num_saving]
                    self.current += num_saving

        return hook

    def register_hook(self):
        start_index = 0
        for module in self.net.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                hook_fn = self.activation_hook(start_index)
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
