"""Shared utilities for Hugging Face text unlearning selection experiments."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn


class TextTensorDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_masks: torch.Tensor,
        targets: torch.Tensor,
    ):
        if not (len(input_ids) == len(attention_masks) == len(targets)):
            raise ValueError("input_ids, attention_masks, and targets must align.")
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.targets = targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return (
            (self.input_ids[index], self.attention_masks[index]),
            self.targets[index],
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


def build_loader(
    input_ids: torch.Tensor,
    attention_masks: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
) -> torch.utils.data.DataLoader:
    dataset = TextTensorDataset(input_ids, attention_masks, targets)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=False,
    )


def build_total_loss(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_masks: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    input_ids = input_ids.to(device, non_blocking=True)
    attention_masks = attention_masks.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    return criterion(model(input_ids, attention_masks), targets)


def _cat_padded(tensors: list[torch.Tensor], pad_value: int = 0) -> torch.Tensor:
    if not tensors:
        raise RuntimeError("Cannot concatenate an empty tensor list.")
    max_length = max(tensor.size(1) for tensor in tensors)
    padded = []
    for tensor in tensors:
        if tensor.size(1) == max_length:
            padded.append(tensor)
            continue
        pad_width = max_length - tensor.size(1)
        padded.append(nn.functional.pad(tensor, (0, pad_width), value=pad_value))
    return torch.cat(padded, dim=0)


def collect_examples(
    dataloader: torch.utils.data.DataLoader,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ids_list: list[torch.Tensor] = []
    masks_list: list[torch.Tensor] = []
    targets_list: list[torch.Tensor] = []
    for (input_ids, attention_masks), targets in dataloader:
        ids_list.append(input_ids)
        masks_list.append(attention_masks)
        targets_list.append(targets)
    if not targets_list:
        raise RuntimeError("No text examples were collected.")
    return (
        _cat_padded(ids_list, pad_value=0),
        _cat_padded(masks_list, pad_value=0),
        torch.cat(targets_list, dim=0),
    )


def collect_target_examples(
    dataloader: torch.utils.data.DataLoader,
    target_label: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _collect_examples(dataloader, target_label, keep_target=True)


def collect_retained_examples(
    dataloader: torch.utils.data.DataLoader,
    target_label: int,
    max_batches: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ids_list: list[torch.Tensor] = []
    masks_list: list[torch.Tensor] = []
    targets_list: list[torch.Tensor] = []
    consumed_batches = 0

    for (input_ids, attention_masks), targets in dataloader:
        mask = targets == target_label
        if not keep_target:
            mask = ~mask
        if torch.any(mask):
            ids_list.append(input_ids[mask])
            masks_list.append(attention_masks[mask])
            targets_list.append(targets[mask])
        consumed_batches += 1
        if max_batches is not None and consumed_batches >= max_batches:
            break

    label_type = "target" if keep_target else "retained"
    if not targets_list:
        raise RuntimeError(f"No {label_type} text examples were collected.")
    return (
        _cat_padded(ids_list, pad_value=0),
        _cat_padded(masks_list, pad_value=0),
        torch.cat(targets_list, dim=0),
    )


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
        for (input_ids, attention_masks), targets in dataloader:
            input_ids = input_ids.to(device, non_blocking=True)
            attention_masks = attention_masks.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(input_ids, attention_masks)
            losses = nn.functional.cross_entropy(outputs, targets, reduction="none")
            predictions = outputs.argmax(dim=1)

            target_mask = targets == target_label
            retain_mask = ~target_mask
            if torch.any(target_mask):
                count = int(target_mask.sum().item())
                target_loss += losses[target_mask].sum().item()
                target_correct += predictions[target_mask].eq(targets[target_mask]).sum().item()
                target_count += count
            if torch.any(retain_mask):
                count = int(retain_mask.sum().item())
                retain_loss += losses[retain_mask].sum().item()
                retain_correct += predictions[retain_mask].eq(targets[retain_mask]).sum().item()
                retain_count += count

    if target_count == 0 or retain_count == 0:
        raise RuntimeError("Could not evaluate both target and retained text splits.")

    self_loss = target_loss / target_count
    retain_loss_value = retain_loss / retain_count
    self_acc = 100.0 * target_correct / target_count
    retain_acc = 100.0 * retain_correct / retain_count
    return _build_metrics(self_loss, self_acc, retain_loss_value, retain_acc)


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
