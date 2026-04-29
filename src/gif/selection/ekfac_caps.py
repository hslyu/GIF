from __future__ import annotations

import numpy as np
import torch

from gif.influence.kfac import (
    _collect_ekfac_batch_stats,
    _ekfac_eigendecomposition,
    _iter_supported_modules,
    _module_grad_matrix,
)

from .base import _ModuleInfo
from .caps import CAPS


def _flatten_preconditioned_like_module(module, preconditioned_matrix: torch.Tensor) -> torch.Tensor:
    if module.bias is None:
        return preconditioned_matrix.reshape(-1)
    return torch.cat(
        [
            preconditioned_matrix[:, :-1].reshape(-1),
            preconditioned_matrix[:, -1].reshape(-1),
        ]
    )


class EKFACCAPS(CAPS):
    """
    EKFAC-scored CAPS:
      - keep the CAPS block construction
      - score each block with an EKFAC quadratic form
      - select top-ranked whole blocks globally until the parameter budget is filled
      - report target-gradient energy coverage of the selected blocks
    """

    def __init__(
        self,
        net,
        ratio,
        lam=1e-6,
        use_attention_head_blocks=True,
    ):
        super().__init__(
            net=net,
            ratio=ratio,
            lam=lam,
            use_attention_head_blocks=use_attention_head_blocks,
        )
        self.gradient_energy_coverage = 0.0
        self.selected_gradient_energy = 0.0
        self.total_gradient_energy = 0.0
        self.block_scores: list[dict[str, float | int | str]] = []

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
        module_dtypes = {
            id(module): next(
                parameter.dtype
                for parameter in module.parameters()
                if parameter.requires_grad
            )
            for module in unique_modules
        }

        target_grads = {
            id(module): torch.zeros(
                module_num_params[id(module)],
                device=device,
                dtype=module_dtypes[id(module)],
            )
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

        retained_inputs_list = []
        retained_targets_list = []
        for inputs, targets in retained_loader:
            retained_inputs_list.append(inputs)
            retained_targets_list.append(targets)

        if len(retained_targets_list) == 0:
            raise RuntimeError("retained_loader is empty.")

        if isinstance(retained_inputs_list[0], (tuple, list)):
            retained_inputs = tuple(
                torch.cat([batch_inputs[i] for batch_inputs in retained_inputs_list], dim=0)
                for i in range(len(retained_inputs_list[0]))
            )
        else:
            retained_inputs = torch.cat(retained_inputs_list, dim=0)
        retained_targets = torch.cat(retained_targets_list, dim=0)

        ekfac_stats = _collect_ekfac_batch_stats(
            self.net,
            _iter_supported_modules(self.net),
            criterion,
            retained_inputs,
            retained_targets,
            device=device,
        )
        stats_by_module = {id(stats.info.module): stats for stats in ekfac_stats.values()}

        scored_blocks = []
        total_gradient_energy = 0.0

        for block in block_specs:
            module = block["module"]
            module_id = id(module)
            stats = stats_by_module.get(module_id)
            if stats is None:
                continue

            target_grad_flat = target_grads[module_id]
            grad_matrix = _module_grad_matrix(module)

            _, activation_basis = _ekfac_eigendecomposition(stats.activation_cov)
            _, gradient_basis = _ekfac_eigendecomposition(stats.gradient_cov)
            grad_kfe = gradient_basis.T @ grad_matrix @ activation_basis
            ekfac_score_matrix = grad_kfe.square() / (stats.corrected_diagonal + self.lam)
            preconditioned_matrix = gradient_basis @ (
                grad_kfe / (stats.corrected_diagonal + self.lam)
            ) @ activation_basis.T
            preconditioned_flat = _flatten_preconditioned_like_module(
                module,
                preconditioned_matrix,
            )

            local_idx = torch.as_tensor(
                block["index_list"],
                device=device,
                dtype=torch.long,
            )
            block_grad = target_grad_flat[local_idx]
            block_preconditioned = preconditioned_flat[local_idx].to(block_grad.dtype)
            block_score = torch.dot(block_grad, block_preconditioned).item()
            block_energy = torch.dot(block_grad, block_grad).item()
            total_gradient_energy += block_energy

            scored_blocks.append(
                {
                    "score": block_score,
                    "energy": block_energy,
                    "block": block,
                    "layer_name": self._module_to_name.get(module_id, str(module_id)),
                    "ekfac_trace_score": ekfac_score_matrix.sum().item(),
                }
            )

        if len(scored_blocks) == 0:
            raise RuntimeError("No EKFAC-selectable blocks were found.")

        total_selectable = sum(item["block"]["num_params"] for item in scored_blocks)
        global_budget = max(1, int(total_selectable * self.ratio))
        min_selectable_block_size = min(
            item["block"]["num_params"] for item in scored_blocks
        )
        global_budget = max(global_budget, min_selectable_block_size)

        scored_blocks.sort(
            key=lambda item: (item["score"], item["energy"]),
            reverse=True,
        )

        self.module_info_list = []
        self.block_scores = []
        selected_uids = set()
        used_global = 0
        selected_gradient_energy = 0.0

        for item in scored_blocks:
            block = item["block"]
            block_uid = (
                block["start_index"],
                tuple(block["index_list"].tolist()),
            )
            if block_uid in selected_uids:
                continue

            block_size = block["num_params"]
            if used_global + block_size > global_budget:
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
            self.block_scores.append(
                {
                    "layer_name": item["layer_name"],
                    "score": float(item["score"]),
                    "energy": float(item["energy"]),
                    "num_params": int(block_size),
                }
            )
            selected_uids.add(block_uid)
            used_global += block_size
            selected_gradient_energy += item["energy"]

        remaining_global = global_budget - used_global
        if remaining_global > 0:
            for item in scored_blocks:
                if remaining_global <= 0:
                    break

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
                self.block_scores.append(
                    {
                        "layer_name": item["layer_name"],
                        "score": float(item["score"]),
                        "energy": float(item["energy"]),
                        "num_params": int(block_size),
                    }
                )
                selected_uids.add(block_uid)
                remaining_global -= block_size
                selected_gradient_energy += item["energy"]

        if len(self.module_info_list) == 0:
            raise RuntimeError(
                "No blocks were selected. Consider increasing ratio."
            )

        self.selected_gradient_energy = float(selected_gradient_energy)
        self.total_gradient_energy = float(total_gradient_energy)
        self.gradient_energy_coverage = float(
            selected_gradient_energy / max(total_gradient_energy, 1e-12)
        )

        self.net.zero_grad(set_to_none=True)
        return self.module_info_list
