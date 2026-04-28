#!/usr/bin/env python3
"""Train MNIST FCN checkpoints across a parameter-scaling grid."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from _mnist_train_common import PROJECT_ROOT, train_mnist_model

from gif.models import FullyConnectedNet


MNIST_FCN_GRID = [
    {"label": "5k", "target_params": 5_000, "hidden_size": 6},
    {"label": "20k", "target_params": 20_000, "hidden_size": 21},
    {"label": "100k", "target_params": 100_000, "hidden_size": 79},
    {"label": "500k", "target_params": 500_000, "hidden_size": 230},
    {"label": "2m", "target_params": 2_000_000, "hidden_size": 514},
    {"label": "10m", "target_params": 10_000_000, "hidden_size": 1226},
    {"label": "20m", "target_params": 20_000_000, "hidden_size": 1760},
]


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MNIST FCN scaling checkpoints for solver stability experiments."
    )
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "scaling",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=[entry["label"] for entry in MNIST_FCN_GRID],
        choices=[entry["label"] for entry in MNIST_FCN_GRID],
        help="Subset of model-size labels to train.",
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
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--dropout-prob", type=float, default=0.1)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = [entry for entry in MNIST_FCN_GRID if entry["label"] in args.sizes]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for entry in selected:
        model = FullyConnectedNet(
            28 * 28,
            int(entry["hidden_size"]),
            10,
            8,
            args.dropout_prob,
        )
        actual_params = count_parameters(model)
        save_path = args.output_dir / f"mnist_fcn_{entry['label']}.pth"
        if args.skip_existing and save_path.is_file():
            print(f"Skipping existing checkpoint: {save_path}")
            continue

        print(
            f"Training MNIST FCN size={entry['label']} "
            f"target_params={entry['target_params']} actual_params={actual_params} "
            f"hidden_size={entry['hidden_size']}"
        )
        train_mnist_model(
            model=model,
            save_path=save_path,
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
            max_train_batches=args.max_train_batches,
            max_val_batches=args.max_val_batches,
            flatten=True,
            validation=True,
            save_trajectory=False,
            save_optimizer_state=False,
            meta={
                "dataset": "mnist",
                "model": "scaling_fcn",
                "scaling_label": entry["label"],
                "target_params": int(entry["target_params"]),
                "actual_params": int(actual_params),
                "hidden_size": int(entry["hidden_size"]),
                "num_layers": 8,
                "dropout_prob": args.dropout_prob,
            },
        )


if __name__ == "__main__":
    main()
