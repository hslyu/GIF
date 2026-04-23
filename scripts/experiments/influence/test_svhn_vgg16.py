#!/usr/bin/env python3
"""Evaluate the SVHN VGG16 checkpoint produced by train_svhn_vgg16.py."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parents[2]
TRAIN_ROOT = PROJECT_ROOT / "scripts" / "train"

for path in (TRAIN_ROOT, PROJECT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from _hf_train_common import run_epoch, set_seed  # noqa: E402
from gif.data.huggingface import create_hf_data_bundle  # noqa: E402
from gif.models import VGG16  # noqa: E402


DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "hf_svhn_vgg16.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a SVHN VGG16 checkpoint on the Hugging Face test split."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--dataset-id", type=str, default=None)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print metrics as JSON instead of a human-readable summary.",
    )
    return parser.parse_args()


def build_model(checkpoint: dict) -> nn.Module:
    return VGG16(
        in_channels=int(checkpoint.get("in_channels", 3) or 3),
        num_classes=int(checkpoint.get("num_classes", 10) or 10),
        classifier_hidden_dim=512,
    )


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> dict:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["net"])
    return checkpoint


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    bundle = create_hf_data_bundle(
        "svhn",
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        validation=True,
        flatten=False,
        seed=args.seed,
        dataset_id=args.dataset_id,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = build_model(checkpoint).to(device)
    checkpoint = load_checkpoint(model, args.checkpoint, device)
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = run_epoch(
        model=model,
        dataloader=bundle.test_loader,
        criterion=criterion,
        device=device,
        optimizer=None,
        max_batches=args.max_test_batches,
        exclude_label=checkpoint.get("exclude_label"),
    )

    metrics = {
        "checkpoint": str(args.checkpoint),
        "dataset": "svhn",
        "model": "vgg16",
        "device": str(device),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "saved_epoch": int(checkpoint.get("epoch", -1)),
        "saved_val_loss": checkpoint.get("val_loss"),
        "saved_val_acc": checkpoint.get("val_acc"),
        "exclude_label": checkpoint.get("exclude_label"),
    }
    if args.json:
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return

    print(f"checkpoint: {metrics['checkpoint']}")
    print(f"device: {metrics['device']}")
    print(f"saved_epoch: {metrics['saved_epoch']}")
    if metrics["saved_val_acc"] is not None:
        print(
            "saved_validation: "
            f"loss={float(metrics['saved_val_loss']):.4f} "
            f"acc={float(metrics['saved_val_acc']):.2f}%"
        )
    print(f"test: loss={metrics['test_loss']:.4f} acc={metrics['test_acc']:.2f}%")


if __name__ == "__main__":
    main()
