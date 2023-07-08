import numpy as np
import torch
from torch import nn


class TopNActivations(nn.Module):
    def __init__(self, net, num_neurons):
        super(TopNActivations, self).__init__()
        self.net = net
        self.hooks = []
        self.num_neurons = num_neurons
        self.activated_neurons = np.zeros(num_neurons, dtype="int16")
        self.current = 0

    def activation_hook(self, start_index):
        def hook(module, input, output):
            batch_mean = torch.mean(output, 0)
            max_idx = torch.argmax(batch_mean)
            weight = module.weight[max_idx]
            _, indices = torch.sort(weight, descending=True)
            indices = indices.detach().cpu().numpy() + start_index

            if self.current + len(indices) > self.num_neurons:
                self.activated_neurons[self.current :] = indices[
                    : self.num_neurons - self.current
                ]
                self.current = self.num_neurons - 1
            else:
                self.activated_neurons[
                    self.current : self.current + len(indices)
                ] = indices
                self.current += len(indices)

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
        self.activated_neurons = np.zeros(self.num_neurons, dtype="int16")
        self.current = 0

    def get_parameters(self):
        return self.activated_neurons
