#!/usr/bin/env python3
"""Train a deep MNIST MLP checkpoint for GIF experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from _mnist_train_common import PROJECT_ROOT, train_mnist_model

from gif.models import FullyConnectedNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a deep FCN MNIST model and save a GIF-compatible checkpoint."
    )
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--save-path",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "mnist_fcn_deep.pth",
    )
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        default=None,
        help="Directory for epoch-wise trajectory checkpoints. Defaults to checkpoints/mnist_fcn_deep/.",
    )
    parser.add_argument(
        "--save-trajectory",
        action="store_true",
        default=True,
        help="Save epoch-wise checkpoints for TracIn trajectory use.",
    )
    parser.add_argument(
        "--no-save-trajectory",
        dest="save_trajectory",
        action="store_false",
        help="Disable epoch-wise trajectory checkpoint saving.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--exclude-label", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--dropout-prob", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_mnist_model(
        model=FullyConnectedNet(
            28 * 28,
            args.hidden_size,
            10,
            args.num_layers,
            args.dropout_prob,
        ),
        save_path=args.save_path,
        trajectory_dir=args.trajectory_dir,
        save_trajectory=args.save_trajectory,
        data_root=args.data_root,
        device=args.device,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        alpha=args.alpha,
        exclude_label=args.exclude_label,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        flatten=True,
        validation=True,
        meta={
            "model": "fcn",
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout_prob": args.dropout_prob,
        },
    )


if __name__ == "__main__":
    main()
