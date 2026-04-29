import numpy as np
import torch
from torch import nn

from .base import Selection, _ModuleInfo


class ReverseCAPS(Selection):
    """
    Reverse-CAPS:
      - uses the same block construction and scoring as CAPS
      - but greedily selects the lowest score-density blocks first
    """

    def __init__(
        self,
        net,
        ratio,
        lam=1e-6,
        use_attention_head_blocks=True,
    ):
        assert 0 < ratio <= 1, "ratio should be in (0, 1]"
        super().__init__()
        self.net = net
        self.ratio = ratio
        self.lam = lam
        self.use_attention_head_blocks = use_attention_head_blocks
        self.module_info_list = []

        self._name_to_module = {}
        self._module_to_name = {}
        self._module_to_parent = {}

    def _is_single_layer(self, module):
        return list(module.children()) == []

    def _build_module_maps(self):
        self._name_to_module = dict(self.net.named_modules())
        self._module_to_name = {}
        self._module_to_parent = {}

        for name, module in self.net.named_modules():
            self._module_to_name[id(module)] = name
            if "." in name:
                parent_name = name.rsplit(".", 1)[0]
                parent = self._name_to_module[parent_name]
            else:
                parent = None
            self._module_to_parent[id(module)] = parent

    def _flatten_module_grads(self, module):
        grad_list = []
        for parameter in module.parameters():
            if not parameter.requires_grad:
                continue
            if parameter.grad is None:
                grad_list.append(torch.zeros_like(parameter).flatten())
            else:
                grad_list.append(parameter.grad.detach().flatten())
        return torch.cat(grad_list) if grad_list else torch.empty(0)

    def _looks_like_attention_projection(self, module_name):
        tokens = (
            "q_proj",
            "k_proj",
            "v_proj",
            "query",
            "key",
            "value",
            "q_lin",
            "k_lin",
            "v_lin",
            ".q.",
            ".k.",
            ".v.",
        )
        return any(tok in module_name for tok in tokens)

    def _get_num_heads_from_parent(self, module):
        parent = self._module_to_parent.get(id(module), None)
        if parent is None:
            return None

        for attr in ("num_heads", "num_attention_heads", "n_head", "n_heads"):
            if hasattr(parent, attr):
                value = getattr(parent, attr)
                if isinstance(value, int) and value > 0:
                    return value
        return None

    def _make_block_dict(
        self,
        module,
        start_index,
        local_idx,
        weight_index_list,
        bias_index_list,
        tag=None,
    ):
        local_idx = np.asarray(local_idx, dtype=int)
        weight_index_list = np.asarray(weight_index_list, dtype=int)
        bias_index_list = np.asarray(bias_index_list, dtype=int)

        return {
            "module": module,
            "start_index": start_index,
            "num_params": len(local_idx),
            "index_list": local_idx,
            "weight_index_list": weight_index_list,
            "bias_index_list": bias_index_list,
            "tag": tag,
        }

    def _make_conv_channel_blocks(self, module, start_index):
        blocks = []
        out_channels = module.weight.shape[0]
        weight_numel = module.weight.numel()
        per_channel_numel = module.weight[0].numel()

        for c in range(out_channels):
            w_start = c * per_channel_numel
            w_end = (c + 1) * per_channel_numel
            weight_index_list = np.arange(w_start, w_end, dtype=int)

            if module.bias is not None:
                bias_local_global = weight_numel + c
                local_idx = np.concatenate(
                    [weight_index_list, np.array([bias_local_global], dtype=int)]
                )
                bias_index_list = np.array([c], dtype=int)
            else:
                local_idx = weight_index_list.copy()
                bias_index_list = np.empty(0, dtype=int)

            blocks.append(
                self._make_block_dict(
                    module=module,
                    start_index=start_index,
                    local_idx=local_idx,
                    weight_index_list=weight_index_list,
                    bias_index_list=bias_index_list,
                    tag=f"conv_channel_{c}",
                )
            )
        return blocks

    def _make_linear_row_or_head_blocks(self, module, start_index):
        blocks = []
        out_features, in_features = module.weight.shape
        weight_numel = module.weight.numel()

        module_name = self._module_to_name.get(id(module), "")
        num_heads = None
        if self.use_attention_head_blocks and self._looks_like_attention_projection(
            module_name
        ):
            num_heads = self._get_num_heads_from_parent(module)

        if num_heads is not None and out_features % num_heads == 0 and num_heads > 0:
            chunk_size = out_features // num_heads
            ranges = [(h * chunk_size, (h + 1) * chunk_size) for h in range(num_heads)]
            tag_prefix = "attn_head"
        else:
            ranges = [(r, r + 1) for r in range(out_features)]
            tag_prefix = "linear_row"

        for block_id, (row_start, row_end) in enumerate(ranges):
            w_start = row_start * in_features
            w_end = row_end * in_features
            weight_index_list = np.arange(w_start, w_end, dtype=int)

            if module.bias is not None:
                bias_rows = np.arange(row_start, row_end, dtype=int)
                local_bias_idx = weight_numel + bias_rows
                local_idx = np.concatenate([weight_index_list, local_bias_idx])
                bias_index_list = bias_rows
            else:
                local_idx = weight_index_list.copy()
                bias_index_list = np.empty(0, dtype=int)

            blocks.append(
                self._make_block_dict(
                    module=module,
                    start_index=start_index,
                    local_idx=local_idx,
                    weight_index_list=weight_index_list,
                    bias_index_list=bias_index_list,
                    tag=f"{tag_prefix}_{block_id}",
                )
            )
        return blocks

    def _iter_blocks(self):
        start_index = 0

        for module in self.net.modules():
            if not self._is_single_layer(module):
                continue

            num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if num_params == 0:
                continue

            if isinstance(module, nn.Conv2d):
                for block in self._make_conv_channel_blocks(module, start_index):
                    yield block
            elif isinstance(module, nn.Linear):
                for block in self._make_linear_row_or_head_blocks(module, start_index):
                    yield block

            start_index += num_params

    def fit(self, target_loader, retained_loader, criterion, device=None):
        self.net.eval()
        self._build_module_maps()

        if device is None:
            try:
                device = next(self.net.parameters()).device
            except StopIteration:
                device = torch.device("cpu")

        block_specs = list(self._iter_blocks())
        if len(block_specs) == 0:
            raise RuntimeError("No selectable blocks were found.")

        unique_modules = []
        seen = set()
        for block in block_specs:
            module = block["module"]
            if id(module) not in seen:
                unique_modules.append(module)
                seen.add(id(module))

        module_num_params = {
            id(module): sum(p.numel() for p in module.parameters() if p.requires_grad)
            for module in unique_modules
        }

        target_grads = {
            id(module): torch.zeros(module_num_params[id(module)], device=device)
            for module in unique_modules
        }
        fisher_diag = {
            id(module): torch.zeros(module_num_params[id(module)], device=device)
            for module in unique_modules
        }

        target_batches = 0
        self.net.zero_grad(set_to_none=True)
        for inputs, targets in target_loader:
            if isinstance(inputs, (tuple, list)):
                inputs = tuple(t.to(device) for t in inputs)
                outputs = self.net(*inputs)
            else:
                inputs = inputs.to(device)
                outputs = self.net(inputs)
            targets = targets.to(device)
            loss = criterion(outputs, targets)
            self.net.zero_grad(set_to_none=True)
            loss.backward()

            for module in unique_modules:
                target_grads[id(module)] += self._flatten_module_grads(module)

            target_batches += 1

        if target_batches == 0:
            raise RuntimeError("target_loader is empty.")

        for module in unique_modules:
            target_grads[id(module)] /= target_batches

        retained_batches = 0
        self.net.zero_grad(set_to_none=True)
        for inputs, targets in retained_loader:
            if isinstance(inputs, (tuple, list)):
                inputs = tuple(t.to(device) for t in inputs)
                outputs = self.net(*inputs)
            else:
                inputs = inputs.to(device)
                outputs = self.net(inputs)
            targets = targets.to(device)
            loss = criterion(outputs, targets)
            self.net.zero_grad(set_to_none=True)
            loss.backward()

            for module in unique_modules:
                grad = self._flatten_module_grads(module)
                fisher_diag[id(module)] += grad * grad

            retained_batches += 1

        if retained_batches == 0:
            raise RuntimeError("retained_loader is empty.")

        for module in unique_modules:
            fisher_diag[id(module)] /= retained_batches

        layer_to_blocks = {}
        layer_to_selectable_params = {}

        for block in block_specs:
            module = block["module"]
            layer_key = id(module)
            layer_name = self._module_to_name.get(layer_key, str(layer_key))
            idx = torch.as_tensor(block["index_list"], device=device, dtype=torch.long)

            block_grad = target_grads[layer_key][idx]
            block_fisher = fisher_diag[layer_key][idx]

            local_scores = (block_grad * block_grad) / (block_fisher + self.lam)
            block_score = local_scores.sum().item()
            score_density = block_score / max(block["num_params"], 1)

            scored_block = {
                "score_density": score_density,
                "block_score": block_score,
                "block": block,
                "layer_key": layer_key,
                "layer_name": layer_name,
            }
            layer_to_blocks.setdefault(layer_key, []).append(scored_block)

        for layer_key, scored_list in layer_to_blocks.items():
            selectable = sum(item["block"]["num_params"] for item in scored_list)
            layer_to_selectable_params[layer_key] = selectable

        total_selectable = sum(layer_to_selectable_params.values())
        if total_selectable == 0:
            raise RuntimeError("No selectable blocks remain.")

        global_budget = max(1, int(total_selectable * self.ratio))

        layer_budget = {}
        layer_fraction = {}
        used_budget_floor = 0

        for layer_key, selectable in layer_to_selectable_params.items():
            raw = global_budget * (selectable / total_selectable)
            alloc = int(np.floor(raw))
            alloc = min(alloc, selectable)
            layer_budget[layer_key] = alloc
            layer_fraction[layer_key] = raw - alloc
            used_budget_floor += alloc

        remaining_budget = global_budget - used_budget_floor
        for layer_key, _ in sorted(
            layer_fraction.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            if remaining_budget <= 0:
                break
            if layer_budget[layer_key] < layer_to_selectable_params[layer_key]:
                layer_budget[layer_key] += 1
                remaining_budget -= 1

        self.module_info_list = []
        selected_uids = set()
        used_global = 0

        for layer_key, scored_list in layer_to_blocks.items():
            scored_list.sort(
                key=lambda x: (x["score_density"], x["block_score"]),
                reverse=False,
            )

            local_budget = layer_budget[layer_key]
            used_local = 0

            for item in scored_list:
                block_score = item["block_score"]
                block = item["block"]

                block_size = block["num_params"]
                if used_local + block_size > local_budget:
                    continue

                block_uid = (
                    block["start_index"],
                    tuple(block["index_list"].tolist()),
                )
                if block_uid in selected_uids:
                    continue

                module_info = _ModuleInfo(
                    module=block["module"],
                    start_index=block["start_index"],
                    num_params=block["num_params"],
                    index_list=block["index_list"],
                    weight_index_list=block["weight_index_list"],
                    bias_index_list=block["bias_index_list"],
                )
                self.module_info_list.append(module_info)
                selected_uids.add(block_uid)
                used_local += block_size
                used_global += block_size

        remaining_global = global_budget - used_global
        if remaining_global > 0:
            for layer_key, scored_list in layer_to_blocks.items():
                if remaining_global <= 0:
                    break

                for item in scored_list:
                    if remaining_global <= 0:
                        break

                    block_score = item["block_score"]
                    block = item["block"]

                    block_uid = (
                        block["start_index"],
                        tuple(block["index_list"].tolist()),
                    )
                    if block_uid in selected_uids:
                        continue

                    block_size = block["num_params"]
                    if block_size > remaining_global:
                        continue

                    module_info = _ModuleInfo(
                        module=block["module"],
                        start_index=block["start_index"],
                        num_params=block["num_params"],
                        index_list=block["index_list"],
                        weight_index_list=block["weight_index_list"],
                        bias_index_list=block["bias_index_list"],
                    )
                    self.module_info_list.append(module_info)
                    selected_uids.add(block_uid)
                    remaining_global -= block_size

        if len(self.module_info_list) == 0:
            raise RuntimeError("No blocks were selected. Consider increasing ratio.")

        self.net.zero_grad(set_to_none=True)
        return self.module_info_list

    def get_parameters(self):
        selected_parameter_indices = np.empty(0, dtype=int)
        for info in self.module_info_list:
            selected_parameter_indices = np.concatenate(
                (selected_parameter_indices, info.index_list + info.start_index)
            )
        return selected_parameter_indices

    def update_network(self, vectorized_influence):
        expected_num_params = sum(info.num_params for info in self.module_info_list)
        assert expected_num_params == len(vectorized_influence), (
            f"length of vectorized_influence {len(vectorized_influence)} "
            f"is not equal to the number of selected parameters {expected_num_params}"
        )

        with torch.no_grad():
            current = 0
            for info in self.module_info_list:
                module = info.module
                change_list = vectorized_influence[current : current + info.num_params]
                current += info.num_params

                weight_change = torch.zeros(
                    module.weight.numel(),
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                if len(info.weight_index_list) > 0:
                    weight_change[info.weight_index_list] = change_list[
                        : len(info.weight_index_list)
                    ]
                module.weight.data += weight_change.view_as(module.weight.data)

                if module.bias is not None:
                    bias_change = torch.zeros(
                        module.bias.numel(),
                        device=module.bias.device,
                        dtype=module.bias.dtype,
                    )
                    if len(info.bias_index_list) > 0:
                        bias_change[info.bias_index_list] = change_list[
                            len(info.weight_index_list) :
                        ]
                    module.bias.data += bias_change.view_as(module.bias.data)
