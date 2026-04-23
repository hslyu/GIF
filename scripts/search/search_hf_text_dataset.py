#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from _hf_text_unlearning_common import PROJECT_ROOT, run_experiment
from gif.models import TextTransformerClassifier


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run influence-based unlearning baselines on Hugging Face text datasets."
    )
    parser.add_argument("--dataset", choices=["newsgroup", "pubmed_rct20k"], required=True)
    parser.add_argument("--dataset-id", type=str, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trajectory-dir", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-target-samples", type=int, default=32)
    parser.add_argument("--num-retain-batches", type=int, default=1)
    parser.add_argument(
        "--schemes",
        nargs="+",
        choices=["gif", "influence", "second_influence", "freeze_influence", "tracin", "hyperinf"],
        default=["gif"],
    )
    parser.add_argument("--param-ratio", type=float, default=0.03)
    parser.add_argument("--caps-lam", type=float, default=1e-5)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--mu", type=float, default=3.0)
    parser.add_argument("--hyperinf-beta-scale", type=float, default=0.9)
    parser.add_argument("--max-iter", type=int, default=4)
    parser.add_argument("--edit-scale", type=float, default=0.01)
    parser.add_argument("--max-text-length", type=int, default=128)
    parser.add_argument("--max-vocab-size", type=int, default=30000)
    parser.add_argument("--min-token-freq", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--text-num-layers", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=512)
    return parser.parse_args()


def build_model(bundle, args):
    return TextTransformerClassifier(
        vocab_size=min(bundle.vocabulary.size, args.max_vocab_size),
        max_len=args.max_text_length,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.text_num_layers,
        dim_feedforward=args.dim_feedforward,
        num_classes=bundle.num_classes,
    )


def main():
    args = parse_args()
    run_experiment(args, model_factory=lambda bundle: build_model(bundle, args))


if __name__ == "__main__":
    main()
