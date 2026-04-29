import numpy as np
import torch
from torch import nn

from .base import _ModuleInfo


def linear_unit_sum_count(value, attention_mask=None):
    detached = value.detach()
    if (
        attention_mask is not None
        and detached.ndim == 3
        and attention_mask.shape == detached.shape[:2]
    ):
        mask = attention_mask.to(device=detached.device, dtype=torch.bool)
        if torch.any(mask):
            return detached[mask].sum(dim=0), int(mask.sum().item())

    reduce_dims = tuple(range(detached.ndim - 1))
    count = 1
    for dim in detached.shape[:-1]:
        count *= int(dim)
    return detached.sum(dim=reduce_dims), count


def conv_unit_sum_count(value):
    detached = value.detach()
    count = int(detached.shape[0] * detached.shape[2] * detached.shape[3])
    return detached.sum(dim=(0, 2, 3)), count


def find_residual_output_modules(net):
    mapping = {}
    for block in net.modules():
        residual_path = None

        # Support common residual branch names across models.
        for residual_attr in ("shortcut", "downsample", "projection"):
            if hasattr(block, residual_attr):
                candidate = getattr(block, residual_attr)
                if isinstance(candidate, nn.Module):
                    residual_path = candidate
                    break

        if residual_path is None:
            continue

        residual_ids = {id(mod) for mod in residual_path.modules()}

        last_conv = None
        for _, module in reversed(list(block.named_modules())):
            if not isinstance(module, nn.Conv2d):
                continue
            if id(module) in residual_ids:
                continue
            last_conv = module
            break

        # Fallback for models that expose residual-conv attributes directly.
        if last_conv is None:
            for name in ("conv3", "conv2", "conv1"):
                candidate = getattr(block, name, None)
                if isinstance(candidate, nn.Conv2d):
                    last_conv = candidate
                    break

        if last_conv is not None:
            mapping[last_conv] = block
    return mapping


def accumulate_unit_scores(score_sums, score_counts, module, value_sum, count):
    key = id(module)
    value_sum = value_sum.detach().cpu()
    if key not in score_sums:
        score_sums[key] = value_sum.clone()
        score_counts[key] = int(count)
    else:
        score_sums[key] += value_sum
        score_counts[key] += int(count)


def make_module_info_from_scores(module, start_index, ratio, score, descending):
    module_size = sum(p.numel() for p in module.parameters() if p.requires_grad)
    num_params = int(module_size * ratio)
    module_info = _ModuleInfo(module, start_index, num_params)
    selected_index_list = np.empty(0, dtype=int)

    if isinstance(module, nn.Linear):
        num_weights_per_output = module.weight.size(1)
    else:
        num_weights_per_output = (
            module.weight.size(1) * module.weight.size(2) * module.weight.size(3)
        )

    if module.bias is not None:
        num_required_indices = num_params // (num_weights_per_output + 1)
        leftover = num_params % (num_weights_per_output + 1)
    else:
        num_required_indices = num_params // num_weights_per_output
        leftover = num_params % num_weights_per_output

    index_list = torch.sort(score, descending=descending, stable=True)[1]
    for index in index_list[:num_required_indices]:
        selected_index_list = np.concatenate(
            (
                selected_index_list,
                np.arange(num_weights_per_output)
                + num_weights_per_output * index.item(),
            )
        )

    if leftover != 0:
        index = index_list[num_required_indices]
        indices = (
            np.arange(leftover)
            + num_weights_per_output * index.item()
        )
        selected_index_list = np.concatenate((selected_index_list, indices))

    module_info.weight_index_list = selected_index_list

    if module.bias is not None:
        module_info.bias_index_list = (
            index_list[:num_required_indices].detach().cpu().numpy()
        )
        selected_index_list = np.concatenate(
            (
                selected_index_list,
                module_info.bias_index_list + module.weight.numel(),
            )
        )
    module_info.index_list = selected_index_list
    return module_info
