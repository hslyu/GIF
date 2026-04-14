import numpy as np
import torch
from torch import nn

from .abstract_selection import Selection, _ModuleInfo


class CAPS(Selection):
    def __init__(self, net, ratio, lam=1e-6, min_curv=1e-12):
        assert 0 < ratio <= 1, "ratio should be in (0, 1]"
        super(CAPS, self).__init__()
        self.net = net
        self.ratio = ratio
        self.lam = lam
        self.min_curv = min_curv
        self.module_info_list = []

    def _is_single_layer(self, module):
        return list(module.children()) == []

    def _iter_blocks(self):
        start_index = 0
        for module in self.net.modules():
            if not self._is_single_layer(module):
                continue

            num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if num_params > 0 and isinstance(module, (nn.Conv2d, nn.Linear)):
                yield module, start_index, num_params
            start_index += num_params

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

    def fit(self, target_loader, retained_loader, criterion, device=None):
        self.net.eval()
        if device is None:
            try:
                device = next(self.net.parameters()).device
            except StopIteration:
                device = torch.device("cpu")

        block_specs = list(self._iter_blocks())
        target_grads = {
            id(module): torch.zeros(num_params, device=device)
            for module, _, num_params in block_specs
        }
        fisher_diag = {
            id(module): torch.zeros(num_params, device=device)
            for module, _, num_params in block_specs
        }

        target_batches = 0
        self.net.zero_grad(set_to_none=True)
        for inputs, targets in target_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            loss = criterion(self.net(inputs), targets)
            self.net.zero_grad(set_to_none=True)
            loss.backward()
            for module, _, _ in block_specs:
                target_grads[id(module)] += self._flatten_module_grads(module)
            target_batches += 1

        if target_batches == 0:
            raise RuntimeError("target_loader is empty.")

        for module, _, _ in block_specs:
            target_grads[id(module)] /= target_batches

        retained_batches = 0
        self.net.zero_grad(set_to_none=True)
        for inputs, targets in retained_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            loss = criterion(self.net(inputs), targets)
            self.net.zero_grad(set_to_none=True)
            loss.backward()
            for module, _, _ in block_specs:
                grad = self._flatten_module_grads(module)
                fisher_diag[id(module)] += grad * grad
            retained_batches += 1

        if retained_batches == 0:
            raise RuntimeError("retained_loader is empty.")

        for module, _, _ in block_specs:
            fisher_diag[id(module)] /= retained_batches

        total_params = sum(
            parameter.numel()
            for parameter in self.net.parameters()
            if parameter.requires_grad
        )
        budget = int(total_params * self.ratio)
        budget = max(1, budget)

        scored_blocks = []
        for module, start_index, num_params in block_specs:
            block_grad = target_grads[id(module)]
            block_fisher = fisher_diag[id(module)]

            if torch.mean(block_fisher) < self.min_curv:
                block_score = float("-inf")
                local_scores = None
            else:
                local_scores = (block_grad * block_grad) / (block_fisher + self.lam)
                block_score = local_scores.sum().item()

            scored_blocks.append(
                (
                    block_score,
                    module,
                    start_index,
                    num_params,
                    local_scores,
                )
            )

        scored_blocks.sort(key=lambda item: item[0], reverse=True)

        self.module_info_list = []
        used = 0
        for score, module, start_index, num_params, local_scores in scored_blocks:
            if used >= budget or score == float("-inf"):
                break

            take = min(num_params, budget - used)
            if take <= 0:
                break

            local_idx = torch.topk(local_scores, k=take, largest=True).indices
            local_idx = local_idx.detach().cpu().numpy().astype(int)

            weight_numel = module.weight.numel()
            weight_mask = local_idx < weight_numel
            weight_index_list = local_idx[weight_mask]

            bias_index_list = np.empty(0, dtype=int)
            if module.bias is not None:
                bias_index_list = local_idx[~weight_mask] - weight_numel

            module_info = _ModuleInfo(
                module=module,
                start_index=start_index,
                num_params=take,
                index_list=local_idx,
                weight_index_list=weight_index_list,
                bias_index_list=bias_index_list,
            )
            self.module_info_list.append(module_info)
            used += take

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
        assert expected_num_params == len(
            vectorized_influence
        ), f"length of vectorized_influence {len(vectorized_influence)} is not equal to the number of selected parameters {expected_num_params}"

        with torch.no_grad():
            current = 0
            for info in self.module_info_list:
                module = info.module
                change_list = vectorized_influence[current : current + info.num_params]
                current += info.num_params

                weight_change = torch.zeros(
                    module.weight.numel(), device=module.weight.device
                )
                if len(info.weight_index_list) > 0:
                    weight_change[info.weight_index_list] = change_list[
                        : len(info.weight_index_list)
                    ]
                module.weight.data += weight_change.view_as(module.weight.data)

                if module.bias is not None:
                    bias_change = torch.zeros(
                        module.bias.numel(), device=module.bias.device
                    )
                    if len(info.bias_index_list) > 0:
                        bias_change[info.bias_index_list] = change_list[
                            len(info.weight_index_list) :
                        ]
                    module.bias.data += bias_change.view_as(module.bias.data)
