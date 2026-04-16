#!/usr/bin/env python3
"""Grid search for MNIST GIF unlearning hyperparameters."""

from __future__ import annotations

import argparse
import itertools
import json
from copy import deepcopy
from pathlib import Path

from run_gif_unlearning_mnist import PROJECT_ROOT, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search for unlearning hyperparameters that preserve retain accuracy while forgetting the target label."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
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
        choices=["caps", "highest_k_gradients"],
        default=["caps", "highest_k_gradients"],
    )
    parser.add_argument(
        "--param-ratios",
        nargs="+",
        type=float,
        default=[0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
    )
    parser.add_argument(
        "--edit-scales",
        nargs="+",
        type=float,
        default=[0.003, 0.01, 0.03, 0.1, 0.3],
    )
    parser.add_argument(
        "--update-scales",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 1.0, 2.0, 4.0],
    )
    parser.add_argument(
        "--caps-lams",
        nargs="+",
        type=float,
        default=[1e-8, 1e-7, 1e-6, 1e-5, 1e-4],
    )
    parser.add_argument("--tols", nargs="+", type=float, default=[1e-9, 1e-8, 1e-7])
    parser.add_argument("--caps-min-curv", type=float, default=1e-12)
    parser.add_argument("--steps-grid", nargs="+", type=float, default=[1.0, 3.0, 5.0])
    parser.add_argument("--max-iters-grid", nargs="+", type=int, default=[20, 30, 50])
    parser.add_argument("--max-update-steps", type=int, default=25)
    parser.add_argument("--min-retain-acc", type=float, default=80.0)
    parser.add_argument(
        "--max-retain-acc-drop",
        type=float,
        default=1.0,
        help="Maximum allowed drop in retain accuracy, in percentage points, relative to the original model.",
    )
    parser.add_argument("--target-self-acc", type=float, default=1.0)
    parser.add_argument("--save-json", type=Path, default=None)
    return parser.parse_args()


def build_run_namespace(search_args: argparse.Namespace, combo: dict[str, float]) -> argparse.Namespace:
    return argparse.Namespace(
        checkpoint=search_args.checkpoint,
        save_path=None,
        data_root=search_args.data_root,
        device=search_args.device,
        seed=search_args.seed,
        target_label=search_args.target_label,
        batch_size=search_args.batch_size,
        num_workers=search_args.num_workers,
        num_target_samples=search_args.num_target_samples,
        num_retain_batches=search_args.num_retain_batches,
        param_ratio=combo["param_ratio"],
        schemes=search_args.schemes,
        caps_lam=combo["caps_lam"],
        caps_min_curv=search_args.caps_min_curv,
        tol=combo["tol"],
        step=combo["step"],
        max_iter=combo["max_iter"],
        edit_scale=combo["edit_scale"],
        update_scale=combo["update_scale"],
        max_update_steps=search_args.max_update_steps,
        min_retain_acc=search_args.min_retain_acc,
        target_self_acc=search_args.target_self_acc,
    )


def enumerate_combinations(search_args: argparse.Namespace) -> list[dict[str, float]]:
    combos = []
    for param_ratio, edit_scale, update_scale, caps_lam, tol, step, max_iter in itertools.product(
        search_args.param_ratios,
        search_args.edit_scales,
        search_args.update_scales,
        search_args.caps_lams,
        search_args.tols,
        search_args.steps_grid,
        search_args.max_iters_grid,
    ):
        combos.append(
            {
                "param_ratio": param_ratio,
                "edit_scale": edit_scale,
                "update_scale": update_scale,
                "caps_lam": caps_lam,
                "tol": tol,
                "step": step,
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
    combinations = enumerate_combinations(search_args)
    print(f"Searching {len(combinations)} combinations")

    all_results = []
    best_by_scheme: dict[str, dict] = {}

    for combo_index, combo in enumerate(combinations, start=1):
        print(
            f"\n[{combo_index}/{len(combinations)}] "
            f"param_ratio={combo['param_ratio']} "
            f"edit_scale={combo['edit_scale']} "
            f"update_scale={combo['update_scale']} "
            f"caps_lam={combo['caps_lam']} "
            f"tol={combo['tol']} "
            f"step={combo['step']} "
            f"max_iter={combo['max_iter']}"
        )
        run_args = build_run_namespace(search_args, combo)
        run_results = run_experiment(run_args)

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
            f"edit_scale={combo['edit_scale']} | "
            f"update_scale={combo['update_scale']} | "
            f"caps_lam={combo['caps_lam']} | "
            f"tol={combo['tol']} | "
            f"step={combo['step']} | "
            f"max_iter={combo['max_iter']} | "
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
                "checkpoint": str(search_args.checkpoint),
                "data_root": str(search_args.data_root),
                "device": search_args.device,
                "target_label": search_args.target_label,
                "schemes": search_args.schemes,
                "param_ratios": search_args.param_ratios,
                "edit_scales": search_args.edit_scales,
                "update_scales": search_args.update_scales,
                "caps_lams": search_args.caps_lams,
                "tols": search_args.tols,
                "steps_grid": search_args.steps_grid,
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
