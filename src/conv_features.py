import torch.nn as nn


class ConvFeatures(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.hook_handle_list = []
        self.forward_result = []
        self.register_hooks()

    def register_hooks(self):
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                hook_handle = module.register_forward_hook(self.forward_hook)
                self.hook_handle_list.append(hook_handle)

    def forward(self, input):
        self.forward_result = []
        self.model(input)
        return self.forward_result

    def forward_hook(self, _, input, output):
        self.forward_result.append(output)
