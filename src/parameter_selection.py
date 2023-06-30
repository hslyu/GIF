import numpy as np
import torch
from torch import nn


class TopNActivations(nn.Module):
    def __init__(self, net, num_neurons):
        super(TopNActivations, self).__init__()
        self.num_neurons = num_neurons
        self.hooks = []
        self.activated_neurons = -np.ones(num_neurons)
        self.net = net

    def activation_hook(self, start_index):
        def hook(module, input, output):
            print(module)
            print(output.shape)
            # for i, activation in enumerate(output):
            #     print(i, activation.shape)

        return hook

    def register_hook(self):
        start_index = 0
        for module in self.net.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                hook_fn = self.activation_hook(start_index)
                hook_handle = module.register_forward_hook(hook_fn)
                start_index += sum(p.numel() for p in module.parameters())
                print(start_index)
                self.hooks.append(hook_handle)
        return self.hooks

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def get_parameters(self):
        return self.activated_neurons
