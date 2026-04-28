"""Shared utilities for image-classification unlearning experiments."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from gif.models import trainable_parameters_to_vector, vector_to_trainable_parameters


class TrainableParameterSelector:
    """Adapter used by influence updates that operate on trainable parameters."""

    def __init__(self, model: nn.Module):
        self.model = model

    def get_parameters(self) -> list[int]:
        return list(range(trainable_parameters_to_vector(self.model).numel()))

    def update_network(self, update: torch.Tensor) -> None:
        base = trainable_parameters_to_vector(self.model).detach()
        vector_to_trainable_parameters(
            base + update.to(base.device, dtype=base.dtype),
            self.model,
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        state_dict = (
            checkpoint.get("net")
            or checkpoint.get("state_dict")
            or checkpoint.get("model_state_dict")
        )
    else:
        state_dict = checkpoint
    if state_dict is None:
        raise KeyError(f"No model state_dict found in checkpoint: {checkpoint_path}")
    model.load_state_dict(state_dict)


def _move_tensor(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if tensor.device == device:
        return tensor
    return tensor.to(device, non_blocking=True)


def build_total_loss(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    inputs = _move_tensor(inputs, device)
    targets = _move_tensor(targets, device)
    return criterion(model(inputs), targets)


def collect_target_examples(
    dataloader: torch.utils.data.DataLoader,
    target_label: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _collect_examples(dataloader, target_label, keep_target=True)


def collect_retained_examples(
    dataloader: torch.utils.data.DataLoader,
    target_label: int,
    max_batches: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _collect_examples(
        dataloader,
        target_label,
        keep_target=False,
        max_batches=max_batches,
    )


def _collect_examples(
    dataloader: torch.utils.data.DataLoader,
    target_label: int,
    *,
    keep_target: bool,
    max_batches: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs_list: list[torch.Tensor] = []
    targets_list: list[torch.Tensor] = []
    consumed_batches = 0

    for inputs, targets in dataloader:
        mask = targets == target_label
        if not keep_target:
            mask = ~mask
        if torch.any(mask):
            inputs_list.append(inputs[mask])
            targets_list.append(targets[mask])
        consumed_batches += 1
        if max_batches is not None and consumed_batches >= max_batches:
            break

    label_type = "target" if keep_target else "retained"
    if not inputs_list:
        raise RuntimeError(f"No {label_type} examples were collected.")
    return torch.cat(inputs_list, dim=0), torch.cat(targets_list, dim=0)


def evaluate_unlearning(
    model: nn.Module,
    target_loader: torch.utils.data.DataLoader,
    retain_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    self_loss, self_acc = _evaluate_loader(model, target_loader, criterion, device)
    retain_loss, retain_acc = _evaluate_loader(model, retain_loader, criterion, device)
    return _build_metrics(self_loss, self_acc, retain_loss, retain_acc)


def _evaluate_loader(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.inference_mode():
        for inputs, targets in dataloader:
            inputs = _move_tensor(inputs, device)
            targets = _move_tensor(targets, device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * targets.size(0)
            total_correct += outputs.argmax(dim=1).eq(targets).sum().item()
            total_examples += targets.size(0)
    if total_examples == 0:
        raise RuntimeError("No examples were processed during evaluation.")
    return total_loss / total_examples, 100.0 * total_correct / total_examples


def _build_metrics(
    self_loss: float,
    self_acc: float,
    retain_loss: float,
    retain_acc: float,
) -> dict[str, float]:
    score = 0.0
    if not (self_acc == 100.0 and retain_acc == 0.0):
        self_acc_norm = self_acc / 100.0
        retain_acc_norm = retain_acc / 100.0
        score = (
            2.0
            * (1.0 - self_acc_norm)
            * retain_acc_norm
            / (1.0 - self_acc_norm + retain_acc_norm)
        )
    return {
        "self_loss": self_loss,
        "self_acc": self_acc,
        "retain_loss": retain_loss,
        "retain_acc": retain_acc,
        "score": score,
    }


def format_metrics(prefix: str, metrics: dict[str, float]) -> str:
    return (
        f"{prefix} "
        f"retain_acc={metrics['retain_acc']:.2f}% | "
        f"self_acc={metrics['self_acc']:.2f}% | "
        f"score={metrics['score']:.4f} | "
        f"retain_loss={metrics['retain_loss']:.4f} | "
        f"self_loss={metrics['self_loss']:.4f}"
    )
