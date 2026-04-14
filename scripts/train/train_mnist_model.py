#!/usr/bin/env python3
"""Train a minimal MNIST classifier for GIF experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from gif.data.mnist import MNISTDataLoader
from gif.models import FullyConnectedNet, LeNet, ResNet18
from gif.regularization import RegularizedLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an MNIST model and save a GIF-compatible checkpoint."
    )
    parser.add_argument("--model", choices=["resnet18", "lenet", "fcn"], default="resnet18")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--save-path", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--exclude-label", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(model_name: str) -> tuple[nn.Module, bool]:
    if model_name == "resnet18":
        return ResNet18(in_channels=1), False
    if model_name == "lenet":
        return LeNet(), False
    return FullyConnectedNet(28 * 28, 128, 10, 4, 0.1), True


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
    batches_used = 0

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
        batches_used += 1

    if total_examples == 0:
        raise RuntimeError("No training examples were processed. Check --exclude-label.")

    return total_loss / total_examples, 100.0 * total_correct / total_examples


def save_checkpoint(
    model: nn.Module,
    save_path: Path,
    epoch: int,
    val_loss: float,
    val_acc: float,
    args: argparse.Namespace,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "net": model.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "model": args.model,
            "alpha": args.alpha,
            "exclude_label": args.exclude_label,
        },
        save_path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    model, flatten = build_model(args.model)
    model = model.to(device)

    data_loader = MNISTDataLoader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        validation=True,
        flatten=flatten,
        root=str(args.data_root),
    )
    train_loader, val_loader, test_loader = data_loader.get_data_loaders()

    base_criterion = nn.CrossEntropyLoss()
    criterion = RegularizedLoss(model, base_criterion, alpha=args.alpha)
    optimizer = SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = float("-inf")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            exclude_label=args.exclude_label,
            max_batches=args.max_train_batches,
        )
        val_loss, val_acc = run_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            exclude_label=args.exclude_label,
            max_batches=args.max_val_batches,
        )
        scheduler.step()

        print(
            f"Epoch {epoch:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, args.save_path, epoch, val_loss, val_acc, args)

    test_loss, test_acc = run_epoch(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        optimizer=None,
        exclude_label=args.exclude_label,
    )
    print(f"Test train-style metric loss={test_loss:.4f} acc={test_acc:.2f}%")
    print(f"Best checkpoint saved to {args.save_path}")


if __name__ == "__main__":
    main()
