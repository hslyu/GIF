import numpy as np
import torch
from torch import nn

from .base import Selection
from .streaming_units import (
    accumulate_unit_scores,
    conv_unit_sum_count,
    find_residual_output_modules,
    linear_unit_sum_count,
    make_module_info_from_scores,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LowestKOutputs(Selection):
    def __init__(self, net, ratio):
        assert 0 < ratio <= 1, "ratio should be in (0, 1]"
        super().__init__()
        self.net = net
        self.ratio = ratio
        self.hook_handle_list = []
        self.module_info_list = []
        self.attention_mask = None
        self.score_sums = {}
        self.score_counts = {}
        self.module_specs = []
        self.finalized = False

    def generate_attention_mask_hook(self):
        def hook(module, input):
            self.attention_mask = None
            if len(input) > 1 and torch.is_tensor(input[1]):
                self.attention_mask = input[1].detach()

        return hook

    def generate_hook(self, start_index):
        def hook(module, input, output):
            if isinstance(module, nn.Linear):
                value_sum, count = linear_unit_sum_count(output, self.attention_mask)
            else:  # isinstance(module, nn.Conv2d):
                value_sum, count = conv_unit_sum_count(output)
            accumulate_unit_scores(
                self.score_sums, self.score_counts, module, value_sum, count
            )

        return hook

    def generate_residual_hook(self, scored_module):
        def hook(module, input, output):
            value_sum, count = conv_unit_sum_count(output)
            accumulate_unit_scores(
                self.score_sums, self.score_counts, scored_module, value_sum, count
            )

        return hook

    def register_hooks(self):
        start_index = 0
        self.module_info_list = []
        self.score_sums = {}
        self.score_counts = {}
        self.module_specs = []
        self.finalized = False
        self.hook_handle_list.append(
            self.net.register_forward_pre_hook(self.generate_attention_mask_hook())
        )
        residual_output_modules = find_residual_output_modules(self.net)
        residual_blocks_registered = set()
        for module in self.net.modules():
            if not self._is_single_layer(module):
                continue

            module_size = sum(p.numel() for p in module.parameters() if p.requires_grad)

            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                self.module_specs.append((module, start_index))
                residual_block = residual_output_modules.get(module)
                if residual_block is not None:
                    block_key = id(residual_block)
                    if block_key not in residual_blocks_registered:
                        hook_fn = self.generate_residual_hook(module)
                        hook_handle = residual_block.register_forward_hook(hook_fn)
                        self.hook_handle_list.append(hook_handle)
                        residual_blocks_registered.add(block_key)
                else:
                    hook_fn = self.generate_hook(start_index)
                    hook_handle = module.register_forward_hook(hook_fn)
                    self.hook_handle_list.append(hook_handle)

            start_index += module_size
        return self.hook_handle_list

    def remove_hooks(self):
        for hook in self.hook_handle_list:
            hook.remove()
        self.hook_handle_list = []
        self.attention_mask = None

    def finalize(self):
        if self.finalized:
            return
        self.module_info_list = []
        for module, start_index in self.module_specs:
            key = id(module)
            if key not in self.score_sums:
                continue
            score = self.score_sums[key] / max(self.score_counts[key], 1)
            self.module_info_list.append(
                make_module_info_from_scores(
                    module, start_index, self.ratio, score, descending=False
                )
            )
        self.finalized = True

    def _is_single_layer(self, module):
        return list(module.children()) == []

    def get_parameters(self):
        self.finalize()
        selected_parameter_indices = np.empty(0, dtype=int)
        for info in self.module_info_list:
            selected_parameter_indices = np.concatenate(
                (selected_parameter_indices, info.index_list + info.start_index)
            )

        return selected_parameter_indices

    def update_network(self, vectorized_influence):
        self.finalize()
        assert sum(info.num_params for info in self.module_info_list) == len(
            vectorized_influence
        ), f"length of vectorized_influence {len(vectorized_influence)} is not equal to the number of seleceted parameters {sum(info.num_params for info in self.module_info_list)}"

        with torch.no_grad():
            current = 0
            for info in self.module_info_list:
                module = info.module
                change_list = vectorized_influence[current : current + info.num_params]
                current += info.num_params

                weight_change = torch.zeros(
                    module.weight.numel(), device=device, dtype=module.weight.dtype
                )
                weight_change[info.weight_index_list] = change_list[
                    : len(info.weight_index_list)
                ]
                module.weight.data += weight_change.view_as(module.weight.data)

                if module.bias is not None:
                    bias_change = torch.zeros(
                        module.bias.numel(), device=device, dtype=module.bias.dtype
                    )
                    bias_change[info.bias_index_list] = change_list[
                        len(info.weight_index_list) :
                    ]
                    module.bias.data += bias_change.view_as(module.bias.data)
