"""Shared utilities for MNIST selection experiments."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn


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


def build_total_loss(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    return criterion(model(inputs.to(device)), targets.to(device))


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
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    target_label: int,
    device: torch.device,
) -> dict[str, float]:
    target_loss = 0.0
    target_correct = 0
    target_count = 0
    retain_loss = 0.0
    retain_correct = 0
    retain_count = 0

    with torch.inference_mode():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            losses = nn.functional.cross_entropy(outputs, targets, reduction="none")
            predictions = outputs.argmax(dim=1)
            target_mask = targets == target_label
            retain_mask = ~target_mask

            if torch.any(target_mask):
                target_loss += losses[target_mask].sum().item()
                target_correct += (
                    predictions[target_mask].eq(targets[target_mask]).sum().item()
                )
                target_count += int(target_mask.sum().item())
            if torch.any(retain_mask):
                retain_loss += losses[retain_mask].sum().item()
                retain_correct += (
                    predictions[retain_mask].eq(targets[retain_mask]).sum().item()
                )
                retain_count += int(retain_mask.sum().item())

    if target_count == 0 or retain_count == 0:
        raise RuntimeError("Could not evaluate both target and retained splits.")

    self_loss = target_loss / target_count
    retain_loss = retain_loss / retain_count
    self_acc = 100.0 * target_correct / target_count
    retain_acc = 100.0 * retain_correct / retain_count
    score = f1_unlearning_score(self_acc, retain_acc)
    return {
        "self_loss": self_loss,
        "self_acc": self_acc,
        "retain_loss": retain_loss,
        "retain_acc": retain_acc,
        "score": score,
    }


def f1_unlearning_score(self_acc: float, retain_acc: float) -> float:
    self_acc /= 100.0
    retain_acc /= 100.0
    if self_acc == 1.0 and retain_acc == 0.0:
        return 0.0
    return 2.0 * (1.0 - self_acc) * retain_acc / (1.0 - self_acc + retain_acc)


def format_metrics(prefix: str, metrics: dict[str, float]) -> str:
    return (
        f"{prefix} "
        f"retain_acc={metrics['retain_acc']:.2f}% | "
        f"self_acc={metrics['self_acc']:.2f}% | "
        f"score={metrics['score']:.4f} | "
        f"retain_loss={metrics['retain_loss']:.4f} | "
        f"self_loss={metrics['self_loss']:.4f}"
    )
