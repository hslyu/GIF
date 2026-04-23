#!/usr/bin/env python3
"""Train image or text models from Hugging Face datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from _hf_train_common import PROJECT_ROOT, train_hf_model

from gif.data.huggingface import create_hf_data_bundle, get_hf_dataset_spec
from gif.models import (
    VGG11,
    VGG16,
    FullyConnectedNet,
    PretrainedTextEncoderClassifier,
    ResNet18,
    ResNet34,
    TextTransformerClassifier,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a model from a Hugging Face dataset and save a GIF-compatible checkpoint."
    )
    parser.add_argument(
        "--dataset",
        choices=["mnist", "cifar10", "svhn", "newsgroup", "pubmed_rct20k"],
        required=True,
    )
    parser.add_argument("--dataset-id", type=str, default=None)
    parser.add_argument(
        "--model",
        choices=[
            "resnet18",
            "resnet34",
            "vgg11",
            "vgg16",
            "fcn",
            "text_transformer",
            "hf_text_encoder",
        ],
        default=None,
    )
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--save-path", type=Path, default=None)
    parser.add_argument("--exclude-label", type=int, default=None)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument(
        "--optimizer",
        choices=["auto", "sgd", "adamw"],
        default="auto",
    )
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        default=None,
        help="Directory for epoch-wise trajectory checkpoints.",
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
        help="Disable trajectory checkpoint saving.",
    )
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--dropout-prob", type=float, default=0.1)
    parser.add_argument("--vocab-size", type=int, default=30000)
    parser.add_argument("--max-text-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--text-num-layers", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=512)
    parser.add_argument("--max-vocab-size", type=int, default=30000)
    parser.add_argument("--min-token-freq", type=int, default=2)
    parser.add_argument(
        "--pretrained-text-model-name",
        type=str,
        default="prajjwal1/bert-tiny",
    )
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    return parser.parse_args()


def resolve_default_model(dataset: str) -> str:
    if dataset in {"newsgroup", "pubmed_rct20k"}:
        return "text_transformer"
    if dataset == "mnist":
        return "fcn"
    return "resnet18"


def resolve_training_defaults(args: argparse.Namespace) -> None:
    if args.model == "text_transformer":
        if args.batch_size is None:
            args.batch_size = 64
        if args.optimizer == "auto":
            args.optimizer = "adamw"
        if args.lr is None:
            args.lr = 3e-4
        if args.weight_decay is None:
            args.weight_decay = 1e-2
        if args.grad_clip_norm is None:
            args.grad_clip_norm = 1.0
    elif args.model == "hf_text_encoder":
        if args.batch_size is None:
            args.batch_size = 32
        if args.optimizer == "auto":
            args.optimizer = "adamw"
        if args.lr is None:
            args.lr = 2e-5
        if args.weight_decay is None:
            args.weight_decay = 1e-2
        if args.grad_clip_norm is None:
            args.grad_clip_norm = 1.0
    else:
        if args.batch_size is None:
            args.batch_size = 256
        if args.optimizer == "auto":
            args.optimizer = "sgd"
        if args.lr is None:
            args.lr = 0.05
        if args.weight_decay is None:
            args.weight_decay = 5e-4


def build_model(args: argparse.Namespace, bundle):
    if args.model == "text_transformer":
        if bundle.vocabulary is None:
            raise ValueError("Text model requires a text dataset bundle.")
        vocab_size = min(bundle.vocabulary.size, args.vocab_size)
        return TextTransformerClassifier(
            vocab_size=vocab_size,
            max_len=args.max_text_length,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.text_num_layers,
            dim_feedforward=args.dim_feedforward,
            num_classes=bundle.num_classes,
            dropout_prob=args.dropout_prob,
        )
    if args.model == "hf_text_encoder":
        return PretrainedTextEncoderClassifier(
            pretrained_model_name=args.pretrained_text_model_name,
            num_classes=bundle.num_classes,
            dropout_prob=args.dropout_prob,
        )
    if args.model == "fcn":
        input_size = bundle.in_channels * bundle.image_size * bundle.image_size
        return FullyConnectedNet(
            input_size=input_size,
            hidden_size=args.hidden_size,
            output_size=bundle.num_classes,
            num_layers=args.num_layers,
            dropout_prob=args.dropout_prob,
        )
    if args.model == "resnet18":
        return ResNet18(in_channels=bundle.in_channels)
    if args.model == "resnet34":
        return ResNet34(in_channels=bundle.in_channels)
    if args.model == "vgg11":
        return VGG11(
            in_channels=bundle.in_channels,
            num_classes=bundle.num_classes,
            classifier_hidden_dim=512,
        )
    if args.model == "vgg16":
        return VGG16(
            in_channels=bundle.in_channels,
            num_classes=bundle.num_classes,
            classifier_hidden_dim=512,
        )
    raise ValueError(f"Unsupported model: {args.model}")


def resolve_save_path(args: argparse.Namespace) -> Path:
    if args.save_path is None:
        resolved = PROJECT_ROOT / "checkpoints" / f"hf_{args.dataset}_{args.model}.pth"
    else:
        resolved = args.save_path

    if args.exclude_label is None:
        return resolved

    suffix = f"_without_{args.exclude_label}"
    if resolved.stem.endswith(suffix):
        return resolved
    return resolved.with_name(f"{resolved.stem}{suffix}{resolved.suffix}")


def main() -> None:
    args = parse_args()
    spec = get_hf_dataset_spec(args.dataset)
    if args.model is None:
        args.model = resolve_default_model(args.dataset)

    if spec.task_type == "text" and args.model not in {
        "text_transformer",
        "hf_text_encoder",
    }:
        raise ValueError(
            "Text datasets require --model text_transformer or --model hf_text_encoder."
        )
    if spec.task_type == "image" and args.model in {
        "text_transformer",
        "hf_text_encoder",
    }:
        raise ValueError("Image datasets do not support text encoder models.")
    resolve_training_defaults(args)
    args.save_path = resolve_save_path(args)

    bundle = create_hf_data_bundle(
        args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        validation=True,
        flatten=args.model == "fcn",
        seed=args.seed,
        dataset_id=args.dataset_id,
        max_text_length=args.max_text_length,
        max_vocab_size=args.max_vocab_size,
        min_token_freq=args.min_token_freq,
        pretrained_text_model_name=(
            args.pretrained_text_model_name if args.model == "hf_text_encoder" else None
        ),
    )
    model = build_model(args, bundle)

    train_hf_model(
        model=model,
        bundle=bundle,
        save_path=args.save_path,
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
        save_trajectory=args.save_trajectory,
        trajectory_dir=args.trajectory_dir,
        grad_clip_norm=args.grad_clip_norm,
        exclude_label=args.exclude_label,
        meta={
            "dataset": args.dataset,
            "dataset_id": args.dataset_id or spec.dataset_id,
            "model": args.model,
            "optimizer": args.optimizer,
            "task_type": spec.task_type,
            "num_classes": bundle.num_classes,
            "in_channels": bundle.in_channels,
            "image_size": bundle.image_size,
            "max_text_length": args.max_text_length
            if args.model in {"text_transformer", "hf_text_encoder"}
            else None,
            "vocab_size": min(bundle.vocabulary.size, args.vocab_size)
            if args.model == "text_transformer"
            else None,
            "d_model": args.d_model if args.model == "text_transformer" else None,
            "nhead": args.nhead if args.model == "text_transformer" else None,
            "text_num_layers": args.text_num_layers
            if args.model == "text_transformer"
            else None,
            "dim_feedforward": args.dim_feedforward
            if args.model == "text_transformer"
            else None,
            "pretrained_text_model_name": args.pretrained_text_model_name
            if args.model == "hf_text_encoder"
            else None,
            "exclude_label": args.exclude_label,
        },
    )


if __name__ == "__main__":
    main()
