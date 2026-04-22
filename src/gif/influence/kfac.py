from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


_ATTENTION_PROJECTION_TOKENS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "query",
    "key",
    "value",
)


@dataclass
class KFACModuleInfo:
    name: str
    module: nn.Module
    kind: str
    role: str
    num_heads: int | None


@dataclass
class KFACFactorStats:
    info: KFACModuleInfo
    activation_cov: torch.Tensor
    gradient_cov: torch.Tensor
    num_samples: int


@dataclass
class EKFACFactorStats(KFACFactorStats):
    corrected_diagonal: torch.Tensor


def _is_supported_kfac_module(module: nn.Module) -> bool:
    return isinstance(module, (nn.Linear, nn.Conv2d))


def _infer_attention_role(name: str, module: nn.Module, parent: nn.Module | None) -> tuple[str, int | None]:
    if not isinstance(module, nn.Linear):
        return "linear", None

    lower_name = name.lower()
    if not any(token in lower_name for token in _ATTENTION_PROJECTION_TOKENS):
        return "linear", None

    num_heads = None
    if parent is not None:
        for attr in ("nhead", "num_heads", "num_attention_heads", "n_heads"):
            value = getattr(parent, attr, None)
            if isinstance(value, int) and value > 0:
                num_heads = value
                break
    return "attention_projection", num_heads


def _iter_supported_modules(model: nn.Module) -> list[KFACModuleInfo]:
    name_to_module = dict(model.named_modules())
    infos: list[KFACModuleInfo] = []
    for name, module in model.named_modules():
        if not _is_supported_kfac_module(module):
            continue
        if not any(parameter.requires_grad for parameter in module.parameters(recurse=False)):
            continue

        parent = None
        if "." in name:
            parent = name_to_module[name.rsplit(".", 1)[0]]

        if isinstance(module, nn.Conv2d):
            infos.append(
                KFACModuleInfo(
                    name=name,
                    module=module,
                    kind="conv2d",
                    role="conv2d",
                    num_heads=None,
                )
            )
            continue

        role, num_heads = _infer_attention_role(name, module, parent)
        infos.append(
            KFACModuleInfo(
                name=name,
                module=module,
                kind="linear",
                role=role,
                num_heads=num_heads,
            )
        )
    return infos


def _move_inputs_to_device(inputs, device: torch.device):
    if isinstance(inputs, (tuple, list)):
        return tuple(value.to(device) for value in inputs)
    return inputs.to(device)


def _prepare_linear_samples(
    module: nn.Linear,
    inputs: torch.Tensor,
    grad_outputs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    activations = inputs.reshape(-1, inputs.shape[-1])
    if module.bias is not None:
        ones = torch.ones(activations.shape[0], 1, device=activations.device, dtype=activations.dtype)
        activations = torch.cat([activations, ones], dim=1)

    gradients = grad_outputs.reshape(-1, grad_outputs.shape[-1])
    return activations, gradients


def _compute_linear_factors(module: nn.Linear, inputs: torch.Tensor, grad_outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    activations, gradients = _prepare_linear_samples(module, inputs, grad_outputs)
    num_samples = activations.shape[0]
    activation_cov = (activations.T @ activations) / max(num_samples, 1)
    gradient_cov = (gradients.T @ gradients) / max(num_samples, 1)
    return activation_cov, gradient_cov, num_samples


def _prepare_conv2d_samples(
    module: nn.Conv2d,
    inputs: torch.Tensor,
    grad_outputs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    unfolded = F.unfold(
        inputs,
        kernel_size=module.kernel_size,
        dilation=module.dilation,
        padding=module.padding,
        stride=module.stride,
    )
    activations = unfolded.transpose(1, 2).reshape(-1, unfolded.shape[1])
    if module.bias is not None:
        ones = torch.ones(activations.shape[0], 1, device=activations.device, dtype=activations.dtype)
        activations = torch.cat([activations, ones], dim=1)

    gradients = grad_outputs.permute(0, 2, 3, 1).reshape(-1, grad_outputs.shape[1])
    return activations, gradients


def _compute_conv2d_factors(module: nn.Conv2d, inputs: torch.Tensor, grad_outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    activations, gradients = _prepare_conv2d_samples(module, inputs, grad_outputs)
    num_samples = activations.shape[0]
    activation_cov = (activations.T @ activations) / max(num_samples, 1)
    gradient_cov = (gradients.T @ gradients) / max(num_samples, 1)
    return activation_cov, gradient_cov, num_samples


def _compute_module_factors(info: KFACModuleInfo, inputs: torch.Tensor, grad_outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    if info.kind == "linear":
        return _compute_linear_factors(info.module, inputs, grad_outputs)
    if info.kind == "conv2d":
        return _compute_conv2d_factors(info.module, inputs, grad_outputs)
    raise ValueError(f"Unsupported KFAC module kind: {info.kind}")


def _prepare_module_samples(
    info: KFACModuleInfo,
    inputs: torch.Tensor,
    grad_outputs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if info.kind == "linear":
        return _prepare_linear_samples(info.module, inputs, grad_outputs)
    if info.kind == "conv2d":
        return _prepare_conv2d_samples(info.module, inputs, grad_outputs)
    raise ValueError(f"Unsupported KFAC module kind: {info.kind}")


def _module_grad_matrix(module: nn.Module) -> torch.Tensor:
    weight_grad = module.weight.grad
    if weight_grad is None:
        weight_matrix = torch.zeros(
            module.weight.shape[0],
            module.weight[0].numel() if isinstance(module, nn.Conv2d) else module.weight.shape[1],
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
    else:
        weight_matrix = weight_grad.reshape(module.weight.shape[0], -1)

    if module.bias is None:
        return weight_matrix

    bias_grad = module.bias.grad
    if bias_grad is None:
        bias_column = torch.zeros(
            module.bias.shape[0],
            1,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
    else:
        bias_column = bias_grad.reshape(-1, 1)
    return torch.cat([weight_matrix, bias_column], dim=1)


def _assign_module_update(module: nn.Module, preconditioned: torch.Tensor) -> list[torch.Tensor]:
    outputs = [preconditioned[:, :-1].reshape_as(module.weight)] if module.bias is not None else [preconditioned.reshape_as(module.weight)]
    if module.bias is not None:
        outputs.append(preconditioned[:, -1].reshape_as(module.bias))
    return outputs


def _flatten_like_model(model: nn.Module, module_updates: dict[int, list[torch.Tensor]]) -> torch.Tensor:
    flat_chunks: list[torch.Tensor] = []
    for module in model.modules():
        params = list(module.parameters(recurse=False))
        if not params:
            continue
        updates = module_updates.get(id(module))
        if updates is None:
            for parameter in params:
                grad = parameter.grad
                if grad is None:
                    flat_chunks.append(torch.zeros_like(parameter).reshape(-1))
                else:
                    flat_chunks.append(grad.detach().reshape(-1))
            continue

        for update in updates:
            flat_chunks.append(update.reshape(-1))

    if not flat_chunks:
        raise RuntimeError("No trainable parameters were found for KFAC update.")
    return torch.cat(flat_chunks)


def _ekfac_eigendecomposition(
    covariance: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = torch.clamp(eigenvalues, min=0.0)
    return eigenvalues, eigenvectors


def _compute_corrected_diagonal(
    info: KFACModuleInfo,
    inputs: torch.Tensor,
    grad_outputs: torch.Tensor,
    activation_basis: torch.Tensor,
    gradient_basis: torch.Tensor,
) -> torch.Tensor:
    activations, gradients = _prepare_module_samples(info, inputs, grad_outputs)
    activation_kfe = activations @ activation_basis
    gradient_kfe = gradients @ gradient_basis
    corrected = torch.einsum(
        "ni,nj->ij",
        gradient_kfe.square(),
        activation_kfe.square(),
    ) / max(activations.shape[0], 1)
    return corrected


def _collect_ekfac_batch_stats(
    model: nn.Module,
    module_infos: list[KFACModuleInfo],
    criterion: nn.Module,
    inputs,
    targets: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, EKFACFactorStats]:
    was_training = model.training
    collector = KFACFactorCollector(model)
    collector.register_hooks()
    model.eval()
    try:
        moved_inputs = _move_inputs_to_device(inputs, device)
        moved_targets = targets.to(device)
        model.zero_grad(set_to_none=True)
        if isinstance(moved_inputs, tuple):
            outputs = model(*moved_inputs)
        else:
            outputs = model(moved_inputs)
        loss = criterion(outputs, moved_targets)
        loss.backward()

        stats: dict[str, EKFACFactorStats] = {}
        for info in module_infos:
            module_id = id(info.module)
            batch_inputs = collector._inputs.get(module_id)
            batch_grad_outputs = collector._grad_outputs.get(module_id)
            if batch_inputs is None or batch_grad_outputs is None:
                continue
            activation_cov, gradient_cov, num_samples = _compute_module_factors(
                info,
                batch_inputs,
                batch_grad_outputs,
            )
            _, activation_basis = _ekfac_eigendecomposition(activation_cov)
            _, gradient_basis = _ekfac_eigendecomposition(gradient_cov)
            corrected_diagonal = _compute_corrected_diagonal(
                info,
                batch_inputs,
                batch_grad_outputs,
                activation_basis,
                gradient_basis,
            )
            stats[info.name] = EKFACFactorStats(
                info=info,
                activation_cov=activation_cov,
                gradient_cov=gradient_cov,
                num_samples=num_samples,
                corrected_diagonal=corrected_diagonal,
            )
        return stats
    finally:
        model.zero_grad(set_to_none=True)
        collector.remove_hooks()
        model.train(was_training)


class KFACFactorCollector:
    def __init__(self, model: nn.Module):
        self.model = model
        self.module_infos = _iter_supported_modules(model)
        self._inputs: dict[int, torch.Tensor] = {}
        self._grad_outputs: dict[int, torch.Tensor] = {}
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []

    def register_hooks(self) -> None:
        self.remove_hooks()
        for info in self.module_infos:
            module = info.module

            def forward_hook(current_module, args, output, module_id=id(module)):
                if not args:
                    return
                tensor = args[0]
                if isinstance(tensor, torch.Tensor):
                    self._inputs[module_id] = tensor.detach()

            def backward_hook(current_module, grad_input, grad_output, module_id=id(module)):
                if not grad_output:
                    return
                tensor = grad_output[0]
                if isinstance(tensor, torch.Tensor):
                    self._grad_outputs[module_id] = tensor.detach()

            self._hooks.append(module.register_forward_hook(forward_hook))
            self._hooks.append(module.register_full_backward_hook(backward_hook))

    def remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._inputs.clear()
        self._grad_outputs.clear()

    def _collect_current_factors(self) -> dict[str, KFACFactorStats]:
        stats: dict[str, KFACFactorStats] = {}
        for info in self.module_infos:
            module_id = id(info.module)
            inputs = self._inputs.get(module_id)
            grad_outputs = self._grad_outputs.get(module_id)
            if inputs is None or grad_outputs is None:
                continue
            activation_cov, gradient_cov, num_samples = _compute_module_factors(
                info, inputs, grad_outputs
            )
            stats[info.name] = KFACFactorStats(
                info=info,
                activation_cov=activation_cov,
                gradient_cov=gradient_cov,
                num_samples=num_samples,
            )
        return stats

    def accumulate_batch(
        self,
        criterion: nn.Module,
        inputs,
        targets: torch.Tensor,
        *,
        device: torch.device | str | None = None,
    ) -> dict[str, KFACFactorStats]:
        if device is None:
            try:
                device = next(self.model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        device = torch.device(device)

        was_training = self.model.training
        self.model.eval()
        self.register_hooks()
        try:
            moved_inputs = _move_inputs_to_device(inputs, device)
            moved_targets = targets.to(device)
            self.model.zero_grad(set_to_none=True)
            if isinstance(moved_inputs, tuple):
                outputs = self.model(*moved_inputs)
            else:
                outputs = self.model(moved_inputs)
            loss = criterion(outputs, moved_targets)
            loss.backward()
            return self._collect_current_factors()
        finally:
            self.model.zero_grad(set_to_none=True)
            self.remove_hooks()
            self.model.train(was_training)

    def accumulate_loader(
        self,
        dataloader: DataLoader,
        criterion: nn.Module,
        *,
        device: torch.device | str | None = None,
        max_batches: int | None = None,
    ) -> dict[str, KFACFactorStats]:
        accumulated: dict[str, KFACFactorStats] = {}
        num_batches = 0
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            num_batches += 1
            inputs, targets = batch
            batch_stats = self.accumulate_batch(
                criterion=criterion,
                inputs=inputs,
                targets=targets,
                device=device,
            )
            for name, stats in batch_stats.items():
                if name not in accumulated:
                    accumulated[name] = KFACFactorStats(
                        info=stats.info,
                        activation_cov=stats.activation_cov.clone(),
                        gradient_cov=stats.gradient_cov.clone(),
                        num_samples=stats.num_samples,
                    )
                    continue

                accumulated[name].activation_cov += stats.activation_cov
                accumulated[name].gradient_cov += stats.gradient_cov
                accumulated[name].num_samples += stats.num_samples

        if num_batches > 0:
            for stats in accumulated.values():
                stats.activation_cov /= num_batches
                stats.gradient_cov /= num_batches
        return accumulated


def kfac_update(
    model: nn.Module,
    retained_inputs,
    retained_targets: torch.Tensor,
    target_inputs,
    target_targets: torch.Tensor,
    criterion: nn.Module,
    *,
    damping: float = 1e-3,
    device: torch.device | str | None = None,
    return_details: bool = False,
):
    if damping < 0:
        raise ValueError("damping must be non-negative.")

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    device = torch.device(device)
    model = model.to(device)

    collector = KFACFactorCollector(model)
    factor_stats = collector.accumulate_batch(
        criterion=criterion,
        inputs=retained_inputs,
        targets=retained_targets,
        device=device,
    )

    moved_target_inputs = _move_inputs_to_device(target_inputs, device)
    moved_target_targets = target_targets.to(device)
    was_training = model.training
    model.eval()
    model.zero_grad(set_to_none=True)
    try:
        if isinstance(moved_target_inputs, tuple):
            outputs = model(*moved_target_inputs)
        else:
            outputs = model(moved_target_inputs)
        target_loss = criterion(outputs, moved_target_targets)
        target_loss.backward()

        module_updates: dict[int, list[torch.Tensor]] = {}
        details: dict[str, dict[str, float | list[int]]] = {}
        for info in collector.module_infos:
            grad_matrix = _module_grad_matrix(info.module)
            stats = factor_stats.get(info.name)
            if stats is None:
                module_updates[id(info.module)] = _assign_module_update(info.module, grad_matrix)
                continue

            a_dim = stats.activation_cov.shape[0]
            g_dim = stats.gradient_cov.shape[0]
            activation_eye = torch.eye(a_dim, device=device, dtype=stats.activation_cov.dtype)
            gradient_eye = torch.eye(g_dim, device=device, dtype=stats.gradient_cov.dtype)
            activation_inv = torch.linalg.inv(stats.activation_cov + damping * activation_eye)
            gradient_inv = torch.linalg.inv(stats.gradient_cov + damping * gradient_eye)
            preconditioned = gradient_inv @ grad_matrix @ activation_inv
            module_updates[id(info.module)] = _assign_module_update(info.module, preconditioned)
            details[info.name] = {
                "activation_shape": list(stats.activation_cov.shape),
                "gradient_shape": list(stats.gradient_cov.shape),
                "role": info.role,
                "num_heads": -1 if info.num_heads is None else info.num_heads,
            }

        update = _flatten_like_model(model, module_updates)
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)

    if return_details:
        return {
            "update": update,
            "details": {
                "damping": float(damping),
                "modules": details,
            },
        }
    return update


class KFACInfluence:
    def compute(
        self,
        model: nn.Module,
        retained_inputs,
        retained_targets: torch.Tensor,
        target_inputs,
        target_targets: torch.Tensor,
        criterion: nn.Module,
        *,
        damping: float = 1e-3,
        device: torch.device | str | None = None,
        return_details: bool = False,
    ):
        return kfac_update(
            model=model,
            retained_inputs=retained_inputs,
            retained_targets=retained_targets,
            target_inputs=target_inputs,
            target_targets=target_targets,
            criterion=criterion,
            damping=damping,
            device=device,
            return_details=return_details,
        )


def ekfac_update(
    model: nn.Module,
    retained_inputs,
    retained_targets: torch.Tensor,
    target_inputs,
    target_targets: torch.Tensor,
    criterion: nn.Module,
    *,
    damping: float = 1e-3,
    device: torch.device | str | None = None,
    return_details: bool = False,
):
    if damping < 0:
        raise ValueError("damping must be non-negative.")

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    device = torch.device(device)
    model = model.to(device)

    module_infos = _iter_supported_modules(model)
    factor_stats = _collect_ekfac_batch_stats(
        model=model,
        module_infos=module_infos,
        criterion=criterion,
        inputs=retained_inputs,
        targets=retained_targets,
        device=device,
    )

    moved_target_inputs = _move_inputs_to_device(target_inputs, device)
    moved_target_targets = target_targets.to(device)
    was_training = model.training
    model.eval()
    model.zero_grad(set_to_none=True)
    try:
        if isinstance(moved_target_inputs, tuple):
            outputs = model(*moved_target_inputs)
        else:
            outputs = model(moved_target_inputs)
        target_loss = criterion(outputs, moved_target_targets)
        target_loss.backward()

        module_updates: dict[int, list[torch.Tensor]] = {}
        details: dict[str, dict[str, float | list[int]]] = {}
        for info in module_infos:
            grad_matrix = _module_grad_matrix(info.module)
            stats = factor_stats.get(info.name)
            if stats is None:
                module_updates[id(info.module)] = _assign_module_update(info.module, grad_matrix)
                continue

            _, activation_basis = _ekfac_eigendecomposition(stats.activation_cov)
            _, gradient_basis = _ekfac_eigendecomposition(stats.gradient_cov)
            diagonal = torch.clamp(stats.corrected_diagonal, min=0.0)
            grad_kfe = gradient_basis.T @ grad_matrix @ activation_basis
            scaled_kfe = grad_kfe / (diagonal + damping)
            preconditioned = gradient_basis @ scaled_kfe @ activation_basis.T
            module_updates[id(info.module)] = _assign_module_update(info.module, preconditioned)
            details[info.name] = {
                "activation_shape": list(stats.activation_cov.shape),
                "gradient_shape": list(stats.gradient_cov.shape),
                "corrected_diagonal_shape": list(stats.corrected_diagonal.shape),
                "role": info.role,
                "num_heads": -1 if info.num_heads is None else info.num_heads,
            }

        update = _flatten_like_model(model, module_updates)
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)

    if return_details:
        return {
            "update": update,
            "details": {
                "damping": float(damping),
                "modules": details,
            },
        }
    return update


class EKFACInfluence:
    def compute(
        self,
        model: nn.Module,
        retained_inputs,
        retained_targets: torch.Tensor,
        target_inputs,
        target_targets: torch.Tensor,
        criterion: nn.Module,
        *,
        damping: float = 1e-3,
        device: torch.device | str | None = None,
        return_details: bool = False,
    ):
        return ekfac_update(
            model=model,
            retained_inputs=retained_inputs,
            retained_targets=retained_targets,
            target_inputs=target_inputs,
            target_targets=target_targets,
            criterion=criterion,
            damping=damping,
            device=device,
            return_details=return_details,
        )


__all__ = [
    "EKFACFactorStats",
    "EKFACInfluence",
    "KFACFactorCollector",
    "KFACFactorStats",
    "KFACInfluence",
    "KFACModuleInfo",
    "ekfac_update",
    "kfac_update",
]
