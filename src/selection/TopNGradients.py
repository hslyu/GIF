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

        self.net = net
        self.num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
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

        #     if isinstance(module, nn.Linear):
        #         # Compute the allotted number of neurons for each rows
        #         num_weights_per_index = module.weight.size(1)
        #         required_rows = quota // num_weights_per_index + 1
        #
        #         batch_mean = torch.mean(torch.abs(grad_output[0]), dim=0)
        #         index_list = torch.sort(batch_mean, descending=True)[1][:required_rows]
        #
        #         # Save the indices of the weights with the highest activations
        #         for index in index_list[:-1]:
        #             self.chosen_param_list[
        #                 self.current : self.current + num_weights_per_index
        #             ] = (
        #                 np.arange(num_weights_per_index)
        #                 + start_index
        #                 + num_weights_per_index * index.item()
        #             )
        #             self.current += num_weights_per_index
        #
        #         # Last index processing
        #         index = index_list[-1]
        #         # quota may not be divisible by num_weights_per_index
        #         num_remaning_indices = quota - num_weights_per_index * (
        #             required_rows - 1
        #         )
        #         weight = module.weight[index]
        #         _, indices = torch.sort(torch.abs(weight), descending=True)
        #         # Add the start position of the module when network is flattened.
        #         indices = (
        #             indices.detach().cpu().numpy()
        #             + start_index  # start of the module
        #             + num_weights_per_index * index.item()  # start of the index
        #         )
        #
        #         self.chosen_param_list[
        #             self.current : self.current + num_remaning_indices
        #         ] = indices[:num_remaning_indices]
        #         self.current += num_remaning_indices
        #
        #     elif isinstance(module, nn.Conv2d):
        #         num_weights_per_kernel = (
        #             module.weight.size(1)
        #             * module.weight.size(2)
        #             * module.weight.size(3)
        #         )  # in_channels (1) * kernel_height (2) * kernel_width (3)
        #         required_kernels = quota // num_weights_per_kernel + 1
        #
        #         batch_mean = torch.mean(
        #             torch.abs(grad_output[0]), (0, 2, 3)
        #         )  # Mean batch (0), height (2), width (3)
        #         index_list = torch.sort(batch_mean, descending=True)[1][
        #             :required_kernels
        #         ]
        #
        #         # Save the indices of the weights with the highest activations
        #         for index in index_list[:-1]:
        #             self.chosen_param_list[
        #                 self.current : self.current + num_weights_per_kernel
        #             ] = (
        #                 np.arange(num_weights_per_kernel)
        #                 + start_index
        #                 + num_weights_per_kernel * index.item()
        #             )
        #             self.current += num_weights_per_kernel
        #
        #         # Get the indices of the weights corresponding to the neurons
        #         index = index_list[-1]
        #         weight = module.weight[index].flatten()
        #         _, indices = torch.sort(torch.abs(weight), descending=True)
        #         # Add the start position of the module when network is flattened.
        #         indices = (
        #             indices.detach().cpu().numpy()
        #             + start_index  # start of the module
        #             + num_weights_per_kernel * index.item()  # start of the index
        #         )
        #
        #         num_remaining_indices = quota - num_weights_per_kernel * (
        #             required_kernels - 1
        #         )
        #         self.chosen_param_list[
        #             self.current : self.current + num_remaining_indices
        #         ] = indices[:num_remaining_indices]
        #         self.current += num_remaining_indices
        #
        # return hook

        return hook

    def register_hooks(self):
        start_index = 0
        for module in self.net.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                hook_fn = self.generate_hook(start_index)
                hook_handle = module.register_full_backward_hook(hook_fn)
                start_index += sum(p.numel() for p in module.parameters())
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
