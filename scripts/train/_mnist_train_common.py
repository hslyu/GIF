from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from gif.data.mnist import MNISTDataLoader
from gif.regularization import RegularizedLoss

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def filter_batch(
    inputs: torch.Tensor, targets: torch.Tensor, exclude_label: int | None
) -> tuple[torch.Tensor, torch.Tensor]:
    if exclude_label is None:
        return inputs, targets
    mask = targets != exclude_label
    return inputs[mask], targets[mask]


def run_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    exclude_label: int | None = None,
    max_batches: int | None = None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        inputs, targets = filter_batch(inputs, targets, exclude_label)
        if targets.numel() == 0:
            continue

        inputs = inputs.to(device)
        targets = targets.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * targets.size(0)
        total_correct += outputs.argmax(dim=1).eq(targets).sum().item()
        total_examples += targets.size(0)

    if total_examples == 0:
        raise RuntimeError("No examples were processed. Check the label filter.")

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


def train_mnist_model(
    model: nn.Module,
    save_path: Path,
    *,
    data_root: Path = PROJECT_ROOT / "data",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 0,
    epochs: int = 20,
    batch_size: int = 256,
    num_workers: int = 4,
    lr: float = 0.05,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    alpha: float = 0.0,
    exclude_label: int | None = None,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    flatten: bool = False,
    validation: bool = True,
    meta: dict[str, object] | None = None,
    save_trajectory: bool = False,
    trajectory_dir: Path | None = None,
    save_optimizer_state: bool = True,
) -> dict[str, float]:
    set_seed(seed)
    device_t = torch.device(device)
    model = model.to(device_t)

    data_loader = MNISTDataLoader(
        batch_size=batch_size,
        num_workers=num_workers,
        validation=validation,
        flatten=flatten,
        root=str(data_root),
    )
    loaders = data_loader.get_data_loaders()
    if validation:
        train_loader, val_loader, test_loader = loaders
    else:
        train_loader, test_loader = loaders
        val_loader = None

    base_criterion = nn.CrossEntropyLoss()
    criterion = RegularizedLoss(model, base_criterion, alpha=alpha)
    optimizer = SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    resolved_trajectory_dir = build_trajectory_dir(save_path, trajectory_dir)

    best_val_acc = float("-inf")
    best_metrics: dict[str, float] | None = None

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            device=device_t,
            optimizer=optimizer,
            exclude_label=exclude_label,
            max_batches=max_train_batches,
        )
        if validation:
            val_loss, val_acc = run_epoch(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device_t,
                optimizer=None,
                exclude_label=exclude_label,
                max_batches=max_val_batches,
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
            "alpha": alpha,
            "exclude_label": exclude_label,
            **(meta or {}),
        }
        if save_optimizer_state:
            meta_payload["optimizer"] = optimizer.state_dict()
        meta_payload["lr"] = current_lr

        if save_trajectory:
            save_checkpoint(
                model=model,
                save_path=trajectory_checkpoint_path(resolved_trajectory_dir, epoch),
                epoch=epoch,
                val_loss=val_loss,
                val_acc=val_acc,
                meta=meta_payload,
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
        dataloader=test_loader,
        criterion=criterion,
        device=device_t,
        optimizer=None,
        exclude_label=exclude_label,
    )
    print(f"Test train-style metric loss={test_loss:.4f} acc={test_acc:.2f}%")
    print(f"Best checkpoint saved to {save_path}")

    return {
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "trajectory_dir": str(resolved_trajectory_dir) if save_trajectory else "",
        **(best_metrics or {}),
    }
