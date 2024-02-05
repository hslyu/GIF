from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCam(nn.Module):
    def __init__(self, model, module_name: str, layer_name_list: list):
        super().__init__()
        self.model = model
        self.module_name = module_name
        self.layer_name_list = layer_name_list
        self.hook_handle_list = []
        self.forward_result_list = []
        self.backward_result_list = []
        self.register_hooks()

    def register_hooks(self):
        for module_name, module in self.model._modules.items():
            if module_name == self.module_name:
                for layer_name, module in module._modules.items():
                    if layer_name in self.layer_name_list:
                        self.hook_handle_list.append(
                            module.register_forward_hook(self.forward_hook)
                        )
                        self.hook_handle_list.append(
                            module.register_backward_hook(self.backward_hook)
                        )

    def remove_hooks(self):
        for handle in self.hook_handle_list:
            handle.remove()

    def forward(self, input, target_index: Optional[int] = None):
        outs = self.model(input)
        outs = outs.squeeze()  # [1, num_classes]  --> [num_classes]

        # if there is no target_index, select the class with the highest score
        if target_index is None:
            target_index = outs.argmax()

        outs[target_index].backward(retain_graph=True)

        mask_list = []
        for i, forward_result in enumerate(self.forward_result_list):
            backward_result = self.backward_result_list[-i - 1]

            a_k = torch.mean(backward_result, dim=(1, 2), keepdim=True)  # [512, 1, 1]
            out = torch.sum(
                a_k * forward_result, dim=0
            ).cpu()  # [512, 7, 7] * [512, 1, 1]
            out = torch.relu(out) / torch.max(out)  # 음수를 없애고, 0 ~ 1 로 scaling # [7, 7]
            out = F.interpolate(
                out.unsqueeze(0).unsqueeze(0), [32, 32], mode="bilinear"
            )
            mask_list.append(out.cpu().detach().squeeze().numpy())

        return mask_list

    def forward_hook(self, module, input, output):
        self.forward_result_list.append(torch.squeeze(output))

    def backward_hook(self, module, grad_input, grad_output):
        self.backward_result_list.append(torch.squeeze(grad_output[0]))
