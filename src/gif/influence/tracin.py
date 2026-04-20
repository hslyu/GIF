from __future__ import annotations

import re
from pathlib import Path

import torch

from gif.influence.common import compute_gradient
from gif.influence.projection import project_subset


EPOCH_CHECKPOINT_PATTERN = re.compile(r"^epoch_(\d+)\.pth$")


def load_tracin_checkpoint_paths(trajectory_dir: Path) -> list[Path]:
    if not trajectory_dir.is_dir():
        raise FileNotFoundError(f"Trajectory directory not found: {trajectory_dir}")

    checkpoint_paths: list[tuple[int, Path]] = []
    for path in trajectory_dir.iterdir():
        if not path.is_file():
            continue
        match = EPOCH_CHECKPOINT_PATTERN.match(path.name)
        if match is None:
            continue
        checkpoint_paths.append((int(match.group(1)), path))

    if not checkpoint_paths:
        raise RuntimeError(
            f"No epoch checkpoints were found in trajectory directory: {trajectory_dir}"
        )

    checkpoint_paths.sort(key=lambda item: item[0])
    return [path for _, path in checkpoint_paths]


def load_tracin_checkpoints(
    trajectory_dir: Path,
    device: torch.device | str = "cpu",
) -> list[dict[str, object]]:
    """Load epoch checkpoints in trajectory order."""
    return [
        torch.load(path, map_location=device)
        for path in load_tracin_checkpoint_paths(trajectory_dir)
    ]


def _trajectory_weight(
    checkpoint: dict[str, object],
    weight_by_lr: bool,
) -> float:
    if not weight_by_lr:
        return 1.0
    value = checkpoint.get("lr", 1.0)
    return float(value)


def _resolve_device(
    model: torch.nn.Module,
    device: torch.device | str | None,
) -> torch.device:
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    return torch.device(device)


def _move_batch_to_device(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    return inputs.to(device), targets.to(device)


def _capture_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def _batch_gradient(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    loss = criterion(model(inputs), targets)
    return compute_gradient(model, loss).detach()


def tracin_score_from_checkpoints(
    model: torch.nn.Module,
    checkpoint_paths: list[Path],
    source_inputs: torch.Tensor,
    source_targets: torch.Tensor,
    target_inputs: torch.Tensor,
    target_targets: torch.Tensor,
    criterion: torch.nn.Module,
    device: torch.device | str | None = None,
    weight_by_lr: bool = True,
    return_details: bool = False,
):
    if len(checkpoint_paths) == 0:
        raise RuntimeError("checkpoint_paths must not be empty.")

    device = _resolve_device(model, device)
    model = model.to(device)
    model.eval()
    source_inputs, source_targets = _move_batch_to_device(
        source_inputs, source_targets, device
    )
    target_inputs, target_targets = _move_batch_to_device(
        target_inputs, target_targets, device
    )
    original_state_dict = _capture_state_dict(model)

    total_score = 0.0
    contributions: list[dict[str, float | str]] = []

    try:
        for checkpoint_path in checkpoint_paths:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["net"])

            source_grad = _batch_gradient(
                model=model,
                criterion=criterion,
                inputs=source_inputs,
                targets=source_targets,
            )
            target_grad = _batch_gradient(
                model=model,
                criterion=criterion,
                inputs=target_inputs,
                targets=target_targets,
            )

            weight = _trajectory_weight(checkpoint, weight_by_lr)
            contribution = weight * torch.dot(source_grad, target_grad).item()
            total_score += contribution

            if return_details:
                contributions.append(
                    {
                        "checkpoint": str(checkpoint_path),
                        "epoch": int(checkpoint.get("epoch", -1)),
                        "lr": weight,
                        "contribution": contribution,
                    }
                )
    finally:
        model.load_state_dict(original_state_dict)

    if return_details:
        return {
            "score": total_score,
            "contributions": contributions,
        }
    return total_score


def tracin_update_from_checkpoints(
    model: torch.nn.Module,
    checkpoint_paths: list[Path],
    target_inputs: torch.Tensor,
    target_targets: torch.Tensor,
    criterion: torch.nn.Module,
    device: torch.device | str | None = None,
    weight_by_lr: bool = True,
    index_list=None,
    return_details: bool = False,
):
    if len(checkpoint_paths) == 0:
        raise RuntimeError("checkpoint_paths must not be empty.")

    device = _resolve_device(model, device)
    model = model.to(device)
    model.eval()
    target_inputs, target_targets = _move_batch_to_device(
        target_inputs, target_targets, device
    )
    original_state_dict = _capture_state_dict(model)

    accumulated_update = None
    contributions: list[dict[str, float | str]] = []

    try:
        for checkpoint_path in checkpoint_paths:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["net"])

            target_grad = _batch_gradient(
                model=model,
                criterion=criterion,
                inputs=target_inputs,
                targets=target_targets,
            )
            weight = _trajectory_weight(checkpoint, weight_by_lr)
            weighted_grad = weight * target_grad

            if accumulated_update is None:
                accumulated_update = torch.zeros_like(weighted_grad)
            accumulated_update += weighted_grad

            if return_details:
                contributions.append(
                    {
                        "checkpoint": str(checkpoint_path),
                        "epoch": int(checkpoint.get("epoch", -1)),
                        "lr": weight,
                        "grad_norm": float(torch.linalg.norm(target_grad).item()),
                    }
                )
    finally:
        model.load_state_dict(original_state_dict)

    if accumulated_update is None:
        raise RuntimeError("TracIn update accumulation failed.")

    # This is a model-edit proxy update built from target gradients along the
    # training trajectory, not the canonical TracIn attribution score itself.
    update = accumulated_update
    if index_list is not None:
        update = project_subset(update, index_list)

    if return_details:
        return {
            "update": update,
            "contributions": contributions,
        }
    return update


class TracIn:
    def load_checkpoints(
        self,
        trajectory_dir: Path,
        device: torch.device | str = "cpu",
    ) -> list[dict[str, object]]:
        return load_tracin_checkpoints(trajectory_dir=trajectory_dir, device=device)

    def compute_scores(
        self,
        model: torch.nn.Module,
        trajectory_dir: Path,
        source_inputs: torch.Tensor,
        source_targets: torch.Tensor,
        target_inputs: torch.Tensor,
        target_targets: torch.Tensor,
        criterion: torch.nn.Module,
        device: torch.device | str | None = None,
        weight_by_lr: bool = True,
        return_details: bool = False,
    ):
        checkpoint_paths = load_tracin_checkpoint_paths(trajectory_dir)
        return tracin_score_from_checkpoints(
            model=model,
            checkpoint_paths=checkpoint_paths,
            source_inputs=source_inputs,
            source_targets=source_targets,
            target_inputs=target_inputs,
            target_targets=target_targets,
            criterion=criterion,
            device=device,
            weight_by_lr=weight_by_lr,
            return_details=return_details,
        )

    def compute_update(
        self,
        model: torch.nn.Module,
        trajectory_dir: Path,
        target_inputs: torch.Tensor,
        target_targets: torch.Tensor,
        criterion: torch.nn.Module,
        device: torch.device | str | None = None,
        weight_by_lr: bool = True,
        index_list=None,
        return_details: bool = False,
    ):
        checkpoint_paths = load_tracin_checkpoint_paths(trajectory_dir)
        return tracin_update_from_checkpoints(
            model=model,
            checkpoint_paths=checkpoint_paths,
            target_inputs=target_inputs,
            target_targets=target_targets,
            criterion=criterion,
            device=device,
            weight_by_lr=weight_by_lr,
            index_list=index_list,
            return_details=return_details,
        )
