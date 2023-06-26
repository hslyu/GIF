import torch


class TopNActivations(torch.nn.Module):
    def __init__(self, net, num_neurons):
        super(TopNActivations, self).__init__()
        self.num_neurons = num_neurons
        self.hooks = []
        self.activated_neurons = []
        self.net = net

    def activation_hook(self, start_index):
        def hook(module, input, output):
            if len(self.activated_neurons) >= self.num_neurons:
                self.activated_neurons[: len(output)] = output
                return

            self.activated_neurons = [
                start_index + i
                for i, activation in enumerate(output)
                if activation >= 0.3
            ]

        return hook

    def register_hook(self):
        start_index = 0
        for layer in self.net.modules():
            print(layer)
            hook_fn = self.activation_hook(start_index)
            hook_handle = layer.register_forward_hook(hook_fn)
            start_index += sum(p.numel() for p in layer.parameters())
            print(start_index)
            self.hooks.append(hook_handle)
        return self.hooks

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def get_parameters(self):
        return self.activated_neurons
