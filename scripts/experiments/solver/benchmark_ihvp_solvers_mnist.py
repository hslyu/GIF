#!/usr/bin/env python3
"""Temporary matrix-free IHVP solver benchmark on the MNIST FCN checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Callable

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gif.data.mnist import MNISTDataLoader  # noqa: E402
from gif.influence import compute_gradient, ekfac_update, hvp, kfac_update  # noqa: E402
from gif.influence.restricted import build_restricted_system  # noqa: E402
from gif.models import FullyConnectedNet  # noqa: E402
from gif.selection import HighestKGradients  # noqa: E402
from gif.solvers import (  # noqa: E402
    cg_inverse,
    hyperinf_inverse,
    lanczos_inverse,
    lissa_inverse,
    p_lissa_inverse,
)

TensorOperator = Callable[[torch.Tensor], torch.Tensor]


class CountedOperator:
    def __init__(self, operator: TensorOperator, *, hvp_per_call: int):
        self.operator = operator
        self.hvp_per_call = hvp_per_call
        self.calls = 0

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.operator(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare p-LiSSA with 100% and 0.1% parameter subsets on the GIF "
            "restricted normal equation against LiSSA, CG, Schulz/HyperInf, "
            "Lanczos, KFAC, and EKFAC on the classical H x = g system."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "mnist_fcn_deep.pth",
    )
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument(
        "--target-batch-size",
        type=int,
        default=1024,
        help="Number of same-label examples for g and HighestKGradients. Defaults to --batch-size.",
    )
    parser.add_argument("--p-lissa-small-ratio", type=float, default=0.1)
    parser.add_argument(
        "--solvers",
        nargs="+",
        default=[
            "p_lissa_full",
            "p_lissa_0p1",
            "lissa",
            "cg",
            "schulz",
            "lanczos",
            "kfac",
            "ekfac",
        ],
        choices=[
            "p_lissa_full",
            "p_lissa_0p1",
            "lissa",
            "cg",
            "schulz",
            "lanczos",
            "kfac",
            "ekfac",
        ],
    )
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--damping", type=float, default=0.00)
    parser.add_argument(
        "--mu",
        type=float,
        default=1,
        help=(
            "Initial p-LiSSA step scale. With --power-iters > 0, p-LiSSA caps "
            "this to roughly 0.9 / lambda_max(H_J^T H_J)."
        ),
    )
    parser.add_argument(
        "--lissa-mu",
        type=float,
        default=1e12,
        help=(
            "Initial classical LiSSA step scale for H x = g. With --power-iters > 0, "
            "LiSSA caps this to roughly 0.9 / lambda_max(H) when the estimate is positive."
        ),
    )
    parser.add_argument("--max-restarts", type=int, default=4)
    parser.add_argument("--power-iters", type=int, default=1)
    parser.add_argument("--schulz-max-iter", type=int, default=10)
    parser.add_argument("--schulz-beta-scale", type=float, default=0.7)
    parser.add_argument("--lanczos-rank", type=int, default=512)
    parser.add_argument("--lanczos-max-iter", type=int, default=800)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device, dtype: torch.dtype):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = FullyConnectedNet(
        28 * 28,
        int(checkpoint.get("hidden_size", 512)),
        10,
        int(checkpoint.get("num_layers", 8)),
        float(checkpoint.get("dropout_prob", 0.1)),
    )
    model.load_state_dict(checkpoint["net"])
    model.to(device=device, dtype=dtype)
    model.eval()
    return model, checkpoint


def load_batches(args: argparse.Namespace, device: torch.device, dtype: torch.dtype):
    torch.manual_seed(args.seed)
    loader = MNISTDataLoader(
        batch_size=args.batch_size,
        num_workers=0,
        validation=True,
        flatten=True,
        root=str(args.data_root),
    )
    train_loader, _, _ = loader.get_data_loaders()
    curvature_inputs, curvature_targets = next(iter(train_loader))

    target_batch_size = args.target_batch_size or args.batch_size
    target_inputs_list: list[torch.Tensor] = []
    target_targets_list: list[torch.Tensor] = []
    collected = 0
    for inputs, targets in train_loader:
        mask = targets == args.target_label
        if not mask.any():
            continue
        selected_inputs = inputs[mask]
        selected_targets = targets[mask]
        remaining = target_batch_size - collected
        target_inputs_list.append(selected_inputs[:remaining])
        target_targets_list.append(selected_targets[:remaining])
        collected += min(remaining, int(selected_targets.numel()))
        if collected >= target_batch_size:
            break

    if collected == 0:
        raise RuntimeError(
            f"No target-label examples found for label={args.target_label}."
        )
    if collected < target_batch_size:
        raise RuntimeError(
            f"Only found {collected} examples for label={args.target_label}; "
            f"requested {target_batch_size}."
        )

    target_inputs = torch.cat(target_inputs_list, dim=0)
    target_targets = torch.cat(target_targets_list, dim=0)
    return (
        curvature_inputs.to(device=device, dtype=dtype),
        curvature_targets.to(device=device),
        target_inputs.to(device=device, dtype=dtype),
        target_targets.to(device=device),
    )


def select_indices(
    model: torch.nn.Module,
    target_inputs: torch.Tensor,
    target_targets: torch.Tensor,
    criterion: torch.nn.Module,
    ratio: float,
    device: torch.device,
) -> torch.Tensor:
    selector = HighestKGradients(model, ratio)
    selector.register_hooks()
    model.zero_grad(set_to_none=True)
    selection_loss = criterion(model(target_inputs), target_targets)
    selection_loss.backward()
    selector.remove_hooks()
    model.zero_grad(set_to_none=True)
    indices = torch.as_tensor(
        selector.get_parameters(), device=device, dtype=torch.long
    )
    if indices.numel() == 0:
        raise RuntimeError(
            f"HighestKGradients selected no parameters for ratio={ratio}."
        )
    return indices.sort().values


def relative_residual(
    a_times: TensorOperator,
    solution: torch.Tensor,
    rhs: torch.Tensor,
    *,
    damping: float = 0.0,
) -> tuple[float, float]:
    rhs_norm = torch.linalg.norm(rhs).item() + 1e-12
    eval_residual = torch.linalg.norm(a_times(solution) - rhs).item() / rhs_norm
    if damping > 0:
        solver_residual = (
            torch.linalg.norm(a_times(solution) + damping * solution - rhs).item()
            / rhs_norm
        )
    else:
        solver_residual = eval_residual
    return eval_residual, solver_residual


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def current_peak_memory_mb(device: torch.device) -> tuple[float | None, float]:
    _, py_peak = tracemalloc.get_traced_memory()
    cuda_peak = None
    if device.type == "cuda":
        cuda_peak = torch.cuda.max_memory_allocated(device) / 1024**2
    return cuda_peak, py_peak / 1024**2


def run_solver(
    name: str,
    base_a_times: TensorOperator,
    rhs: torch.Tensor,
    system_name: str,
    hvp_per_a_times: int,
    args: argparse.Namespace,
    device: torch.device,
    *,
    model: torch.nn.Module | None = None,
    curvature_inputs: torch.Tensor | None = None,
    curvature_targets: torch.Tensor | None = None,
    target_inputs: torch.Tensor | None = None,
    target_targets: torch.Tensor | None = None,
    criterion: torch.nn.Module | None = None,
) -> dict[str, object]:
    counted = CountedOperator(base_a_times, hvp_per_call=hvp_per_a_times)
    details: dict[str, object] = {}
    failed = False
    error = ""
    solution = torch.full_like(rhs, float("nan"))

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    tracemalloc.start()
    synchronize(device)
    start = time.perf_counter()

    try:
        if name in {"p_lissa_full", "p_lissa_0p1"}:
            solution = p_lissa_inverse(
                a_times=counted,
                rhs=rhs,
                damping=0.0,
                mu=args.mu,
                tol=args.tol,
                max_iter=args.max_iter,
                max_restarts=args.max_restarts,
                power_iter_steps=args.power_iters,
            )
        elif name == "lissa":
            result = lissa_inverse(
                a_times=counted,
                rhs=rhs,
                damping=args.damping,
                mu=args.lissa_mu,
                tol=args.tol,
                max_iter=args.max_iter,
                max_restarts=args.max_restarts,
                power_iter_steps=args.power_iters,
                return_details=True,
            )
            solution = result["solution"]
            details = result["details"]
        elif name == "cg":
            result = cg_inverse(
                a_times=counted,
                rhs=rhs,
                damping=args.damping,
                tol=args.tol,
                max_iter=args.max_iter,
                return_details=True,
            )
            solution = result["solution"]
            details = result["details"]
        elif name == "schulz":
            result = hyperinf_inverse(
                a_times=counted,
                rhs=rhs,
                beta_scale=args.schulz_beta_scale,
                tol=args.tol,
                max_iter=args.schulz_max_iter,
                power_iter_steps=args.power_iters,
                return_details=True,
            )
            solution = result["solution"]
            details = result["details"]
        elif name == "lanczos":
            result = lanczos_inverse(
                a_times=counted,
                rhs=rhs,
                rank=args.lanczos_rank,
                damping=args.damping,
                tol=args.tol,
                max_iter=args.lanczos_max_iter,
                return_details=True,
            )
            solution = result["solution"]
            details = result["details"]
        elif name == "kfac":
            if (
                model is None
                or curvature_inputs is None
                or curvature_targets is None
                or target_inputs is None
                or target_targets is None
                or criterion is None
            ):
                raise RuntimeError("KFAC benchmark requires model and batch inputs.")
            result = kfac_update(
                model=model,
                retained_inputs=curvature_inputs,
                retained_targets=curvature_targets,
                target_inputs=target_inputs,
                target_targets=target_targets,
                criterion=criterion,
                damping=args.damping,
                device=device,
                return_details=True,
            )
            solution = result["update"]
            details = result["details"]
        elif name == "ekfac":
            if (
                model is None
                or curvature_inputs is None
                or curvature_targets is None
                or target_inputs is None
                or target_targets is None
                or criterion is None
            ):
                raise RuntimeError("EKFAC benchmark requires model and batch inputs.")
            result = ekfac_update(
                model=model,
                retained_inputs=curvature_inputs,
                retained_targets=curvature_targets,
                target_inputs=target_inputs,
                target_targets=target_targets,
                criterion=criterion,
                damping=args.damping,
                device=device,
                return_details=True,
            )
            solution = result["update"]
            details = result["details"]
        else:
            raise ValueError(f"Unknown solver: {name}")
    except Exception as exc:  # noqa: BLE001 - benchmark should record failures.
        failed = True
        error = f"{type(exc).__name__}: {exc}"

    synchronize(device)
    elapsed = time.perf_counter() - start
    cuda_peak_mb, python_peak_mb = current_peak_memory_mb(device)
    tracemalloc.stop()

    eval_residual = math.nan
    solver_residual = math.nan
    update_norm = math.nan
    finite = bool(torch.isfinite(solution).all().item())
    if not failed and finite:
        before_eval_calls = counted.calls
        eval_residual, solver_residual = relative_residual(
            counted,
            solution,
            rhs,
            damping=args.damping
            if name
            in {
                "lissa",
                "cg",
                "lanczos",
                "kfac",
                "ekfac",
            }
            else 0.0,
        )
        eval_calls = counted.calls - before_eval_calls
        update_norm = torch.linalg.norm(solution).item()
    else:
        eval_calls = 0

    solve_a_times_calls = counted.calls - eval_calls
    return {
        "solver": name,
        "system": system_name,
        "failed": failed,
        "error": error,
        "finite": finite,
        "eval_residual": eval_residual,
        "solver_residual": solver_residual,
        "time_sec": elapsed,
        "cuda_peak_mb": cuda_peak_mb,
        "python_peak_mb": python_peak_mb,
        "a_times_calls": solve_a_times_calls,
        "hvp_per_a_times": hvp_per_a_times,
        "hvp_equiv_calls": hvp_per_a_times * solve_a_times_calls,
        "update_norm": update_norm,
        "requested_mu": (
            args.mu
            if name in {"p_lissa_full", "p_lissa_0p1"}
            else args.lissa_mu
            if name == "lissa"
            else None
        ),
        "iterations": details.get("iterations", None),
        "converged": details.get("converged", None),
        "details": details,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.float32

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model, checkpoint = load_model(args.checkpoint, device, dtype)
    curvature_inputs, curvature_targets, target_inputs, target_targets = load_batches(
        args,
        device,
        dtype,
    )
    criterion = torch.nn.CrossEntropyLoss()

    curvature_outputs = model(curvature_inputs)
    total_loss = criterion(curvature_outputs, curvature_targets)
    target_outputs = model(target_inputs)
    target_loss = criterion(target_outputs, target_targets)
    g_full = compute_gradient(model, target_loss, retain_graph=True)
    small_index_list = select_indices(
        model=model,
        target_inputs=target_inputs,
        target_targets=target_targets,
        criterion=criterion,
        ratio=args.p_lissa_small_ratio,
        device=device,
    )
    full_index_list = torch.arange(g_full.numel(), device=device, dtype=torch.long)
    full_restricted_rhs, full_restricted_a_times = build_restricted_system(
        model,
        total_loss,
        g_full,
        full_index_list,
    )
    small_restricted_rhs, small_restricted_a_times = build_restricted_system(
        model,
        total_loss,
        g_full,
        small_index_list,
    )

    def classical_h_times(value: torch.Tensor) -> torch.Tensor:
        return hvp(model, total_loss, value)

    classical_rhs = g_full

    print(
        f"p_lissa_full selected_params={int(full_index_list.numel())}; "
        f"p_lissa_0p1 selection=highest_k_gradients "
        f"ratio={args.p_lissa_small_ratio:.4f} "
        f"selected_params={int(small_index_list.numel())}"
    )
    print(
        f"curvature_batch_size={args.batch_size} target_label={args.target_label} "
        f"target_batch_size={int(target_targets.numel())}"
    )
    print(
        "systems: p_lissa_full/p_lissa_0p1 solve H_J^T H_J x = H_J^T g; "
        "lissa/cg/schulz/lanczos solve H x = g"
    )

    rows = []
    for solver in args.solvers:
        if solver == "p_lissa_full":
            rhs = full_restricted_rhs
            a_times = full_restricted_a_times
            system_name = "restricted_normal_full"
            hvp_per_a_times = 2
        elif solver == "p_lissa_0p1":
            rhs = small_restricted_rhs
            a_times = small_restricted_a_times
            system_name = "restricted_normal"
            hvp_per_a_times = 2
        else:
            rhs = classical_rhs
            a_times = classical_h_times
            system_name = "classical_hessian"
            hvp_per_a_times = 1

        row = run_solver(
            solver,
            a_times,
            rhs,
            system_name,
            hvp_per_a_times,
            args,
            device,
            model=model,
            curvature_inputs=curvature_inputs,
            curvature_targets=curvature_targets,
            target_inputs=target_inputs,
            target_targets=target_targets,
            criterion=criterion,
        )
        rows.append(row)
        mem_cuda = (
            "None" if row["cuda_peak_mb"] is None else f"{row['cuda_peak_mb']:.2f}"
        )
        print(
            f"{solver:12s} "
            f"eval_res={row['eval_residual']:.3e} "
            f"solver_res={row['solver_residual']:.3e} "
            f"time={row['time_sec']:.3f}s "
            f"mem_cuda={mem_cuda}MB A_calls={row['a_times_calls']} "
            f"HVP~={row['hvp_equiv_calls']}"
        )
        if row["error"]:
            print(f"  error: {row['error']}")

    payload = {
        "config": {
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_val_acc": checkpoint.get("val_acc"),
            "device": str(device),
            "dtype": "float32",
            "batch_size": args.batch_size,
            "target_label": args.target_label,
            "target_batch_size": int(target_targets.numel()),
            "p_lissa_full_ratio": 1.0,
            "p_lissa_full_selected_params": int(full_index_list.numel()),
            "p_lissa_small_ratio": args.p_lissa_small_ratio,
            "p_lissa_small_selected_params": int(small_index_list.numel()),
            "p_lissa_small_selection": "highest_k_gradients",
            "p_lissa_full_rhs_norm": torch.linalg.norm(full_restricted_rhs).item(),
            "p_lissa_small_rhs_norm": torch.linalg.norm(small_restricted_rhs).item(),
            "classical_rhs_norm": torch.linalg.norm(classical_rhs).item(),
            "p_lissa_full_system": "restricted_normal_full",
            "p_lissa_small_system": "restricted_normal",
            "other_solver_system": "classical_hessian",
            "damping": args.damping,
            "tol": args.tol,
        },
        "results": rows,
    }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix == ".json":
            args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        elif args.output.suffix == ".csv":
            fieldnames = [
                "solver",
                "system",
                "failed",
                "finite",
                "eval_residual",
                "solver_residual",
                "time_sec",
                "cuda_peak_mb",
                "python_peak_mb",
                "a_times_calls",
                "hvp_per_a_times",
                "hvp_equiv_calls",
                "update_norm",
                "requested_mu",
                "iterations",
                "converged",
                "error",
            ]
            with args.output.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow({key: row.get(key) for key in fieldnames})
        else:
            raise ValueError("--output must end with .json or .csv")


if __name__ == "__main__":
    main()
