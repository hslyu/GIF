#!/usr/bin/env python3
"""Search GIF unlearning hyperparameters for trained MNIST models."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from copy import deepcopy
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from _mnist_unlearning_common import PROJECT_ROOT, run_experiment  # noqa: E402
from gif.models import FullyConnectedNet, ResNet18, ResNet34  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search for unlearning hyperparameters that preserve retain accuracy while forgetting the target label."
    )
    parser.add_argument(
        "--model",
        choices=["resnet18", "resnet34", "fcn"],
        default="resnet34",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--trajectory-dir", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-target-samples", type=int, default=256)
    parser.add_argument("--num-retain-batches", type=int, default=1)
    parser.add_argument(
        "--schemes",
        nargs="+",
        choices=["caps", "highest_k_gradients", "tracin"],
        default=["caps"],
    )
    parser.add_argument(
        "--param-ratios",
        nargs="+",
        type=float,
        default=[0.03, 0.05, 0.07, 0.1],
    )
    parser.add_argument("--edit-scale", type=float, default=0.02)
    parser.add_argument("--caps-lam", type=float, default=1e-5)
    parser.add_argument(
        "--tol-grid",
        nargs="+",
        type=float,
        default=[1e-3, 1e-4, 1e-5, 1e-6],
    )
    parser.add_argument("--caps-min-curv", type=float, default=1e-12)
    parser.add_argument("--mu", type=float, default=3.0)
    parser.add_argument(
        "--max-iters-grid", nargs="+", type=int, default=[100, 200, 300]
    )
    parser.add_argument("--max-update-steps", type=int, default=1000)
    parser.add_argument("--min-retain-acc", type=float, default=98.5)
    parser.add_argument(
        "--max-retain-acc-drop",
        type=float,
        default=1.0,
        help="Maximum allowed drop in retain accuracy, in percentage points, relative to the original model.",
    )
    parser.add_argument(
        "--target-self-acc",
        type=float,
        default=0.2,
        help="Stop a run when target self accuracy drops below this percentage value. Example: 0.2 means 0.2%%.",
    )
    parser.add_argument("--save-json", type=Path, default=None)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--dropout-prob", type=float, default=0.1)
    return parser.parse_args()


def build_model(args: argparse.Namespace):
    if args.model == "resnet18":
        return ResNet18(in_channels=1), False
    if args.model == "resnet34":
        return ResNet34(in_channels=1), False
    if args.model == "fcn":
        return (
            FullyConnectedNet(
                28 * 28,
                args.hidden_size,
                10,
                args.num_layers,
                args.dropout_prob,
            ),
            True,
        )
    raise ValueError(f"Unsupported model: {args.model}")


def build_checkpoint_path(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        return args.checkpoint
    if args.model == "fcn":
        return PROJECT_ROOT / "checkpoints" / "mnist_fcn_deep.pth"
    return PROJECT_ROOT / "checkpoints" / f"mnist_{args.model}.pth"


def build_trajectory_path(args: argparse.Namespace) -> Path | None:
    if args.trajectory_dir is not None:
        return args.trajectory_dir
    checkpoint_path = build_checkpoint_path(args)
    return checkpoint_path.parent / checkpoint_path.stem


def build_run_namespace(
    search_args: argparse.Namespace, combo: dict[str, float]
) -> argparse.Namespace:
    return argparse.Namespace(
        checkpoint=build_checkpoint_path(search_args),
        save_path=None,
        data_root=search_args.data_root,
        device=search_args.device,
        seed=search_args.seed,
        target_label=search_args.target_label,
        batch_size=search_args.batch_size,
        num_workers=search_args.num_workers,
        num_target_samples=search_args.num_target_samples,
        num_retain_batches=search_args.num_retain_batches,
        trajectory_dir=build_trajectory_path(search_args),
        param_ratio=combo["param_ratio"],
        schemes=search_args.schemes,
        caps_lam=search_args.caps_lam,
        caps_min_curv=search_args.caps_min_curv,
        tol=combo["tol"],
        mu=search_args.mu,
        max_iter=combo["max_iter"],
        edit_scale=search_args.edit_scale,
        max_update_steps=search_args.max_update_steps,
        min_retain_acc=search_args.min_retain_acc,
        target_self_acc=search_args.target_self_acc,
    )


def enumerate_combinations(search_args: argparse.Namespace) -> list[dict[str, float]]:
    combos = []
    for param_ratio, tol, max_iter in itertools.product(
        search_args.param_ratios,
        search_args.tol_grid,
        search_args.max_iters_grid,
    ):
        combos.append(
            {
                "param_ratio": param_ratio,
                "tol": tol,
                "max_iter": max_iter,
            }
        )
    return combos


def scheme_rank_tuple(
    metrics: dict[str, float],
    min_retain_acc: float,
    max_retain_acc_drop: float,
) -> tuple[bool, bool, float, float, float, float]:
    retain_drop = metrics["retain_acc_drop"]
    return (
        metrics["retain_acc"] >= min_retain_acc,
        retain_drop <= max_retain_acc_drop,
        -retain_drop,
        metrics["score"],
        -metrics["self_acc"],
        metrics["retain_acc"],
    )


def main() -> None:
    search_args = parse_args()
    model, flatten = build_model(search_args)
    combinations = enumerate_combinations(search_args)
    print(f"Searching {len(combinations)} combinations")

    all_results = []
    best_by_scheme: dict[str, dict] = {}

    for combo_index, combo in enumerate(combinations, start=1):
        print(
            f"\n[{combo_index}/{len(combinations)}] "
            f"param_ratio={combo['param_ratio']} "
            f"tol={combo['tol']} "
            f"max_iter={combo['max_iter']} "
            f"edit_scale={search_args.edit_scale} "
            f"caps_lam={search_args.caps_lam} "
            f"mu={search_args.mu}"
        )
        run_args = build_run_namespace(search_args, combo)
        run_results = run_experiment(
            run_args,
            model_factory=lambda: build_model(search_args)[0],
            flatten=flatten,
        )

        record = {"combo": deepcopy(combo), "results": run_results}
        all_results.append(record)

        for scheme, metrics in run_results.items():
            candidate = {
                "combo": deepcopy(combo),
                "metrics": deepcopy(metrics),
            }
            previous = best_by_scheme.get(scheme)
            candidate_rank = scheme_rank_tuple(
                candidate["metrics"],
                search_args.min_retain_acc,
                search_args.max_retain_acc_drop,
            )
            previous_rank = (
                None
                if previous is None
                else scheme_rank_tuple(
                    previous["metrics"],
                    search_args.min_retain_acc,
                    search_args.max_retain_acc_drop,
                )
            )
            if previous is None or candidate_rank > previous_rank:
                best_by_scheme[scheme] = candidate

    print("\nBest combinations")
    for scheme in search_args.schemes:
        best = best_by_scheme[scheme]
        combo = best["combo"]
        metrics = best["metrics"]
        print(
            f"{scheme:>20} | "
            f"param_ratio={combo['param_ratio']} | "
            f"tol={combo['tol']} | "
            f"max_iter={combo['max_iter']} | "
            f"edit_scale={search_args.edit_scale} | "
            f"caps_lam={search_args.caps_lam} | "
            f"mu={search_args.mu} | "
            f"retain_drop={metrics['retain_acc_drop']:.2f} | "
            f"orig_retain_acc={metrics['before_retain_acc']:.2f}% | "
            f"retain_acc={metrics['retain_acc']:.2f}% | "
            f"self_acc={metrics['self_acc']:.2f}% | "
            f"score={metrics['score']:.4f}"
        )

    if search_args.save_json is not None:
        search_args.save_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "search_args": {
                "checkpoint": str(build_checkpoint_path(search_args)),
                "data_root": str(search_args.data_root),
                "device": search_args.device,
                "target_label": search_args.target_label,
                "model": search_args.model,
                "schemes": search_args.schemes,
                "param_ratios": search_args.param_ratios,
                "edit_scale": search_args.edit_scale,
                "caps_lam": search_args.caps_lam,
                "tol_grid": search_args.tol_grid,
                "mu": search_args.mu,
                "max_iters_grid": search_args.max_iters_grid,
                "min_retain_acc": search_args.min_retain_acc,
                "max_retain_acc_drop": search_args.max_retain_acc_drop,
            },
            "best_by_scheme": best_by_scheme,
            "all_results": all_results,
        }
        search_args.save_json.write_text(json.dumps(payload, indent=2))
        print(f"Saved search results to {search_args.save_json}")


if __name__ == "__main__":
    main()
