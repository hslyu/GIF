from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from gif.data.huggingface import HFDataBundle
from gif.regularization import RegularizedLoss

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_inputs_to_device(inputs, device: torch.device):
    if isinstance(inputs, (tuple, list)):
        return tuple(tensor.to(device) for tensor in inputs)
    return inputs.to(device)


def forward_model(model: nn.Module, inputs):
    if isinstance(inputs, (tuple, list)):
        return model(*inputs)
    return model(inputs)


def filter_batch(inputs, targets: torch.Tensor, exclude_label: int | None):
    if exclude_label is None:
        return inputs, targets

    mask = targets != exclude_label
    if isinstance(inputs, (tuple, list)):
        filtered_inputs = tuple(tensor[mask] for tensor in inputs)
    else:
        filtered_inputs = inputs[mask]
    return filtered_inputs, targets[mask]


def run_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None = None,
    grad_clip_norm: float | None = None,
    exclude_label: int | None = None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        inputs, targets = batch
        inputs, targets = filter_batch(inputs, targets, exclude_label)
        if targets.numel() == 0:
            continue
        inputs = move_inputs_to_device(inputs, device)
        targets = targets.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            outputs = forward_model(model, inputs)
            loss = criterion(outputs, targets)
            if is_train:
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

        total_loss += loss.item() * targets.size(0)
        total_correct += outputs.argmax(dim=1).eq(targets).sum().item()
        total_examples += targets.size(0)

    if total_examples == 0:
        raise RuntimeError("No examples were processed.")

    return total_loss / total_examples, 100.0 * total_correct / total_examples


def save_checkpoint(
    model: nn.Module,
    save_path: Path,
    epoch: int,
    val_loss: float,
    val_acc: float,
    meta: dict[str, object],
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "net": model.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "val_acc": val_acc,
            **meta,
        },
        save_path,
    )


def build_trajectory_dir(save_path: Path, trajectory_dir: Path | None = None) -> Path:
    if trajectory_dir is not None:
        return trajectory_dir
    return save_path.parent / save_path.stem


def trajectory_checkpoint_path(trajectory_dir: Path, epoch: int) -> Path:
    return trajectory_dir / f"epoch_{epoch:03d}.pth"


def prune_trajectory_after_epoch(trajectory_dir: Path, best_epoch: int) -> None:
    if not trajectory_dir.is_dir():
        return
    for path in trajectory_dir.glob("epoch_*.pth"):
        try:
            epoch = int(path.stem.split("_")[-1])
        except ValueError:
            continue
        if epoch > best_epoch:
            path.unlink()


def train_hf_model(
    model: nn.Module,
    bundle: HFDataBundle,
    save_path: Path,
    *,
    device: str,
    seed: int,
    epochs: int,
    optimizer_name: str,
    lr: float,
    momentum: float,
    weight_decay: float,
    alpha: float,
    max_train_batches: int | None,
    max_val_batches: int | None,
    max_test_batches: int | None,
    save_trajectory: bool = False,
    trajectory_dir: Path | None = None,
    grad_clip_norm: float | None = None,
    meta: dict[str, object] | None = None,
    exclude_label: int | None = None,
) -> dict[str, float]:
    set_seed(seed)
    device_t = torch.device(device)
    model = model.to(device_t)

    base_criterion = nn.CrossEntropyLoss()
    criterion = RegularizedLoss(model, base_criterion, alpha=alpha)
    if optimizer_name == "adamw":
        optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "sgd":
        optimizer = SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    resolved_trajectory_dir = build_trajectory_dir(save_path, trajectory_dir)

    best_val_acc = float("-inf")
    best_metrics: dict[str, float] | None = None

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(
            model=model,
            dataloader=bundle.train_loader,
            criterion=criterion,
            device=device_t,
            optimizer=optimizer,
            max_batches=max_train_batches,
            grad_clip_norm=grad_clip_norm,
            exclude_label=exclude_label,
        )
        if bundle.val_loader is not None:
            val_loss, val_acc = run_epoch(
                model=model,
                dataloader=bundle.val_loader,
                criterion=criterion,
                device=device_t,
                optimizer=None,
                max_batches=max_val_batches,
                exclude_label=exclude_label,
            )
        else:
            val_loss, val_acc = train_loss, train_acc
        current_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step()

        print(
            f"Epoch {epoch:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}%"
        )

        meta_payload = {
            **(meta or {}),
            "alpha": alpha,
            "lr": current_lr,
            "exclude_label": exclude_label,
        }
        if save_trajectory:
            save_checkpoint(
                model,
                trajectory_checkpoint_path(resolved_trajectory_dir, epoch),
                epoch,
                val_loss,
                val_acc,
                meta_payload,
            )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, save_path, epoch, val_loss, val_acc, meta_payload)
            best_metrics = {
                "epoch": float(epoch),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
            }

    test_loss, test_acc = run_epoch(
        model=model,
        dataloader=bundle.test_loader,
        criterion=criterion,
        device=device_t,
        optimizer=None,
        max_batches=max_test_batches,
        exclude_label=exclude_label,
    )
    print(f"Test train-style metric loss={test_loss:.4f} acc={test_acc:.2f}%")
    print(f"Best checkpoint saved to {save_path}")

    if save_trajectory and best_metrics is not None:
        prune_trajectory_after_epoch(
            resolved_trajectory_dir,
            int(best_metrics["epoch"]),
        )

    return {
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "trajectory_dir": str(resolved_trajectory_dir) if save_trajectory else "",
        **(best_metrics or {}),
    }
