#!/usr/bin/env python3
"""Train CIFAR10 ScaledResNet18 checkpoints across a parameter-scaling grid."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from _hf_train_common import PROJECT_ROOT, train_hf_model

from gif.data.huggingface import create_hf_data_bundle


CIFAR10_RESNET_GRID = [
    {"label": "25k", "target_params": 25_000, "base_width": 3},
    {"label": "100k", "target_params": 100_000, "base_width": 6},
    {"label": "500k", "target_params": 500_000, "base_width": 13},
    {"label": "2m", "target_params": 2_000_000, "base_width": 27},
    {"label": "5m", "target_params": 5_000_000, "base_width": 43},
    {"label": "10m", "target_params": 10_000_000, "base_width": 61},
    {"label": "20m", "target_params": 20_000_000, "base_width": 86},
]


class ScaledBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ScaledResNet18(nn.Module):
    def __init__(self, base_width: int, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        self.base_width = base_width
        self.in_planes = base_width
        self.conv1 = nn.Conv2d(
            in_channels,
            base_width,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(base_width)
        self.layer1 = self._make_layer(base_width, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(base_width * 2, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(base_width * 4, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(base_width * 8, num_blocks=2, stride=2)
        self.linear = nn.Linear(base_width * 8, num_classes)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for block_stride in strides:
            layers.append(ScaledBasicBlock(self.in_planes, planes, block_stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return self.linear(out)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train CIFAR10 ScaledResNet18 scaling checkpoints for solver stability "
            "experiments."
        )
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
        default=[entry["label"] for entry in CIFAR10_RESNET_GRID],
        choices=[entry["label"] for entry in CIFAR10_RESNET_GRID],
        help="Subset of model-size labels to train.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = [entry for entry in CIFAR10_RESNET_GRID if entry["label"] in args.sizes]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bundle = create_hf_data_bundle(
        "cifar10",
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        validation=True,
        flatten=False,
        seed=args.seed,
    )

    for entry in selected:
        model = ScaledResNet18(
            base_width=int(entry["base_width"]),
            in_channels=bundle.in_channels,
            num_classes=bundle.num_classes,
        )
        actual_params = count_parameters(model)
        save_path = args.output_dir / f"cifar10_scaled_resnet18_{entry['label']}.pth"
        if args.skip_existing and save_path.is_file():
            print(f"Skipping existing checkpoint: {save_path}")
            continue

        print(
            f"Training CIFAR10 ScaledResNet18 size={entry['label']} "
            f"target_params={entry['target_params']} actual_params={actual_params} "
            f"base_width={entry['base_width']}"
        )
        train_hf_model(
            model=model,
            bundle=bundle,
            save_path=save_path,
            device=args.device,
            seed=args.seed,
            epochs=args.epochs,
            optimizer_name=args.optimizer,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            alpha=args.alpha,
            max_train_batches=args.max_train_batches,
            max_val_batches=args.max_val_batches,
            max_test_batches=args.max_test_batches,
            save_trajectory=False,
            trajectory_dir=None,
            grad_clip_norm=args.grad_clip_norm,
            exclude_label=None,
            meta={
                "dataset": "cifar10",
                "dataset_id": "tanganke/cifar10",
                "model": "scaled_resnet18",
                "scaling_label": entry["label"],
                "target_params": int(entry["target_params"]),
                "actual_params": int(actual_params),
                "base_width": int(entry["base_width"]),
                "num_classes": bundle.num_classes,
                "in_channels": bundle.in_channels,
                "image_size": bundle.image_size,
                "optimizer": args.optimizer,
            },
        )


if __name__ == "__main__":
    main()
