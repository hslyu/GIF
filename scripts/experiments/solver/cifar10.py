#!/usr/bin/env python3
"""
Estimate CIFAR-10 IHVP solver accuracy across scaling checkpoints.

This script mirrors scripts/experiments/solver/mnist.py for CIFAR-10.
It scans model-size checkpoints, keeps p-LiSSA damping-free, tries damping
candidates independently for non-p-LiSSA solvers, and records the smallest
damping value that produces a finite update and finite residual.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import subprocess
import sys
import time
import tracemalloc
from contextlib import contextmanager
from copy import copy
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gif.data.huggingface import create_hf_data_bundle  # noqa: E402
from gif.influence import compute_gradient, ekfac_update, hvp, kfac_update  # noqa: E402
from gif.influence.projection import project_subset  # noqa: E402
from gif.selection import HighestKGradients  # noqa: E402
from gif.solvers import (  # noqa: E402
    cg_inverse,
    hyperinf_inverse,
    lanczos_inverse,
    lissa_inverse,
    p_lissa_inverse,
)

TensorOperator = Callable[[torch.Tensor], torch.Tensor]

CIFAR10_SIZES = ["25k", "100k", "500k", "2m", "5m", "10m", "20m"]
SOLVERS = [
    "p_lissa_full",
    "p_lissa_0p1",
    "lissa",
    "cg",
    "schulz",
    "lanczos",
    "kfac",
    "ekfac",
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


class SolverTimeoutError(TimeoutError):
    pass


def _raise_solver_timeout(signum, frame) -> None:  # noqa: ARG001
    raise SolverTimeoutError("solver time limit exceeded")


@contextmanager
def solver_time_limit(seconds: float | None):
    if seconds is None or seconds <= 0:
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_solver_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


class CountedOperator:
    def __init__(self, operator: TensorOperator, *, hvp_per_call: int):
        self.operator = operator
        self.hvp_per_call = hvp_per_call
        self.calls = 0

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.operator(value)


def parse_damping_candidates(value: str) -> list[float]:
    candidates = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        candidate = float(item)
        if candidate < 0:
            raise argparse.ArgumentTypeError("damping candidates must be non-negative")
        candidates.append(candidate)
    if not candidates:
        raise argparse.ArgumentTypeError("at least one damping candidate is required")
    return sorted(dict.fromkeys(candidates))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate IHVP solver residuals across CIFAR-10 ScaledResNet18 "
            "scaling checkpoints, "
            "with p-LiSSA damping fixed to zero and non-p-LiSSA damping chosen "
            "as the smallest candidate that avoids NaN/Inf outputs."
        )
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "scaling",
    )
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "scripts" / "experiments" / "solver" / "results",
        help="Directory where aggregate and per-seed result files are written.",
    )
    parser.add_argument(
        "--sizes", nargs="+", default=CIFAR10_SIZES, choices=CIFAR10_SIZES
    )
    parser.add_argument("--solvers", nargs="+", default=SOLVERS, choices=SOLVERS)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run selected solvers even when matching result rows already exist.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed. Used directly unless --seeds is provided.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Explicit seed list. Overrides --seed/--num-seeds.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Run seeds seed, seed+1, ..., seed+num_seeds-1.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument("--target-batch-size", type=int, default=1024)
    parser.add_argument("--p-lissa-small-ratio", type=float, default=0.1)
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--lissa-mu", type=float, default=1e12)
    parser.add_argument("--max-restarts", type=int, default=4)
    parser.add_argument("--power-iters", type=int, default=1)
    parser.add_argument("--schulz-max-iter", type=int, default=10)
    parser.add_argument("--schulz-beta-scale", type=float, default=0.1)
    parser.add_argument("--lanczos-rank", type=int, default=256)
    parser.add_argument("--lanczos-max-iter", type=int, default=800)
    parser.add_argument(
        "--hvp-num-batches",
        type=int,
        default=2,
        help="Use this many DataLoader batches for the retained curvature HVP set.",
    )
    parser.add_argument(
        "--damping-candidates",
        type=parse_damping_candidates,
        default=parse_damping_candidates("0,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1"),
        help="Default comma-separated damping grid for non-p-LiSSA solvers.",
    )
    parser.add_argument(
        "--lissa-damping-candidates",
        type=parse_damping_candidates,
        default=parse_damping_candidates("1e-2,1e-1,1"),
    )
    parser.add_argument(
        "--cg-damping-candidates",
        type=parse_damping_candidates,
        default=parse_damping_candidates("1e-4,1e-3,1e-2"),
    )
    parser.add_argument(
        "--no-cg-cgnr-fallback",
        action="store_true",
        help=(
            "Disable CGNR fallback. By default, if standard CG fails because "
            "H+dI is not positive definite, CG retries on the normal equation "
            "to obtain a finite least-squares residual."
        ),
    )
    parser.add_argument(
        "--cg-cgnr-damping",
        type=float,
        default=1e-12,
        help="Tiny ridge added inside the CGNR fallback normal equation.",
    )
    parser.add_argument("--lanczos-damping-candidates", type=parse_damping_candidates)
    parser.add_argument("--kfac-damping-candidates", type=parse_damping_candidates)
    parser.add_argument("--ekfac-damping-candidates", type=parse_damping_candidates)
    parser.add_argument(
        "--no-kfac-scale-calibration",
        action="store_true",
        help=(
            "Disable scalar calibration for KFAC/EKFAC. By default the script "
            "rescales their update by the least-squares alpha that minimizes "
            "||H(alpha x) - g|| for the evaluated Hessian system."
        ),
    )
    parser.add_argument(
        "--no-lissa-scale-calibration",
        action="store_true",
        help=(
            "Disable scalar calibration for LiSSA. By default the script rescales "
            "the LiSSA update by the least-squares alpha that minimizes "
            "||H(alpha x) - g|| for the evaluated Hessian system."
        ),
    )
    parser.add_argument(
        "--time-limit-sec",
        type=float,
        default=600.0,
        help=(
            "Per solver/model runtime budget. A run that finishes after this "
            "budget is marked timeout and the experiment continues by default."
        ),
    )
    parser.add_argument(
        "--keep-running-after-timeout",
        action="store_true",
        help="Deprecated; timeout runs continue by default.",
    )
    parser.add_argument(
        "--stop-after-timeout",
        action="store_true",
        help="Stop the experiment after a solver exceeds the time budget.",
    )
    parser.add_argument(
        "--wait-for-idle-gpu",
        action="store_true",
        help=(
            "Before each timed solver run, wait until nvidia-smi reports low GPU "
            "utilization and little external GPU memory use."
        ),
    )
    parser.add_argument(
        "--idle-gpu-util-threshold",
        type=float,
        default=5.0,
        help="Maximum allowed GPU utilization percent for --wait-for-idle-gpu.",
    )
    parser.add_argument(
        "--idle-external-memory-mb",
        type=float,
        default=256.0,
        help=(
            "Maximum GPU memory, in MB, used by processes other than this script "
            "for --wait-for-idle-gpu."
        ),
    )
    parser.add_argument(
        "--idle-check-interval-sec",
        type=float,
        default=5.0,
        help="Polling interval for --wait-for-idle-gpu.",
    )
    parser.add_argument(
        "--idle-timeout-sec",
        type=float,
        default=0.0,
        help=(
            "Maximum idle wait. Use 0 to wait indefinitely. If exceeded, the "
            "experiment raises RuntimeError."
        ),
    )
    return parser.parse_args()


def resolve_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds is not None:
        if not args.seeds:
            raise ValueError("--seeds must contain at least one seed")
        return list(dict.fromkeys(args.seeds))
    if args.num_seeds < 1:
        raise ValueError("--num-seeds must be at least 1")
    return [args.seed + offset for offset in range(args.num_seeds)]


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def load_model(checkpoint_path: Path, device: torch.device, dtype: torch.dtype):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = ScaledResNet18(
        base_width=int(checkpoint["base_width"]),
        in_channels=int(checkpoint.get("in_channels", 3)),
        num_classes=int(checkpoint.get("num_classes", 10)),
    )
    model.load_state_dict(checkpoint["net"])
    model.to(device=device, dtype=dtype)
    model.eval()
    return model, checkpoint


def load_batches(args: argparse.Namespace, device: torch.device, dtype: torch.dtype):
    bundle = create_hf_data_bundle(
        "cifar10",
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        validation=True,
        flatten=False,
        seed=args.seed,
    )
    train_loader = bundle.train_loader
    curvature_inputs_list: list[torch.Tensor] = []
    curvature_targets_list: list[torch.Tensor] = []
    train_iter = iter(train_loader)
    for _ in range(args.hvp_num_batches):
        try:
            batch_inputs, batch_targets = next(train_iter)
        except StopIteration as exc:
            raise RuntimeError(
                f"Only found {len(curvature_targets_list)} curvature batches; "
                f"requested {args.hvp_num_batches}."
            ) from exc
        curvature_inputs_list.append(batch_inputs)
        curvature_targets_list.append(batch_targets)
    curvature_inputs = torch.cat(curvature_inputs_list, dim=0)
    curvature_targets = torch.cat(curvature_targets_list, dim=0)

    target_inputs_list: list[torch.Tensor] = []
    target_targets_list: list[torch.Tensor] = []
    collected = 0
    for inputs, targets in train_loader:
        mask = targets == args.target_label
        if not mask.any():
            continue
        remaining = args.target_batch_size - collected
        target_inputs_list.append(inputs[mask][:remaining])
        target_targets_list.append(targets[mask][:remaining])
        collected += min(remaining, int(mask.sum().item()))
        if collected >= args.target_batch_size:
            break

    if collected < args.target_batch_size:
        raise RuntimeError(
            f"Only found {collected} examples for label={args.target_label}; "
            f"requested {args.target_batch_size}."
        )

    target_inputs = torch.cat(target_inputs_list, dim=0)
    target_targets = torch.cat(target_targets_list, dim=0)
    return (
        train_loader,
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
    loss = criterion(model(target_inputs), target_targets)
    loss.backward()
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


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def current_peak_memory_mb(device: torch.device) -> tuple[float | None, float]:
    _, py_peak = tracemalloc.get_traced_memory()
    cuda_peak = None
    if device.type == "cuda":
        cuda_peak = torch.cuda.max_memory_allocated(device) / 1024**2
    return cuda_peak, py_peak / 1024**2


def nvidia_smi_device_id(device: torch.device) -> str:
    visible_index = device.index
    if visible_index is None:
        visible_index = torch.cuda.current_device()

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        entries = [entry.strip() for entry in visible_devices.split(",")]
        if 0 <= visible_index < len(entries) and entries[visible_index]:
            return entries[visible_index]
    return str(visible_index)


def run_nvidia_smi(device_id: str, query: str) -> list[str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--id",
            device_id,
            f"--query-{query}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def gpu_utilization_percent(device_id: str) -> float:
    lines = run_nvidia_smi(device_id, "gpu=utilization.gpu")
    return float(lines[0].split(",")[0].strip()) if lines else math.nan


def external_gpu_memory_mb(device_id: str) -> float:
    current_pid = os.getpid()
    total = 0.0
    for line in run_nvidia_smi(device_id, "compute-apps=pid,used_memory"):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
            used_mb = float(parts[1])
        except ValueError:
            continue
        if pid != current_pid:
            total += used_mb
    return total


def wait_for_idle_gpu(args: argparse.Namespace, device: torch.device) -> float:
    if not args.wait_for_idle_gpu or device.type != "cuda":
        return 0.0

    device_id = nvidia_smi_device_id(device)
    start = time.perf_counter()
    printed_wait = False
    while True:
        try:
            util_pct = gpu_utilization_percent(device_id)
            external_mb = external_gpu_memory_mb(device_id)
        except (FileNotFoundError, subprocess.CalledProcessError, ValueError) as exc:
            raise RuntimeError(
                "--wait-for-idle-gpu requires working nvidia-smi queries"
            ) from exc

        if (
            util_pct <= args.idle_gpu_util_threshold
            and external_mb <= args.idle_external_memory_mb
        ):
            waited = time.perf_counter() - start
            if printed_wait:
                print(f"    idle GPU acquired after {waited:.1f}s")
            return waited

        elapsed = time.perf_counter() - start
        if args.idle_timeout_sec > 0 and elapsed > args.idle_timeout_sec:
            raise RuntimeError(
                "GPU did not become idle within "
                f"{args.idle_timeout_sec:.1f}s: util={util_pct:.1f}%, "
                f"external_mem={external_mb:.1f}MB"
            )
        if not printed_wait:
            print(
                "    waiting for idle GPU "
                f"(util={util_pct:.1f}%, external_mem={external_mb:.1f}MB)"
            )
            printed_wait = True
        time.sleep(max(args.idle_check_interval_sec, 0.1))


def relative_residual(
    a_times: TensorOperator,
    solution: torch.Tensor,
    rhs: torch.Tensor,
    *,
    damping: float,
) -> tuple[float, float]:
    rhs_norm = torch.linalg.norm(rhs).item() + 1e-12
    Ax = a_times(solution)
    eval_residual = torch.linalg.norm(Ax - rhs).item() / rhs_norm
    solver_residual = (
        torch.linalg.norm(Ax + damping * solution - rhs).item() / rhs_norm
        if damping > 0
        else eval_residual
    )
    return eval_residual, solver_residual


def make_batched_hvp_fn(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: torch.nn.Module,
    device: torch.device,
    num_batches: int,
) -> TensorOperator:
    if num_batches < 1:
        raise ValueError("--hvp-num-batches must be at least 1")

    input_chunks = torch.chunk(inputs, num_batches, dim=0)
    target_chunks = torch.chunk(targets, num_batches, dim=0)
    total_examples = int(targets.size(0))
    if total_examples == 0:
        raise RuntimeError("Cannot build batched HVP from an empty curvature batch.")

    def hvp_fn(vector: torch.Tensor) -> torch.Tensor:
        accumulated = None
        for batch_inputs, batch_targets in zip(
            input_chunks, target_chunks, strict=True
        ):
            batch_inputs = batch_inputs.to(device=device)
            batch_targets = batch_targets.to(device=device)
            batch_loss = criterion(model(batch_inputs), batch_targets)
            batch_hvp = hvp(model, batch_loss, vector)
            weight = batch_targets.size(0) / total_examples
            if accumulated is None:
                accumulated = batch_hvp * weight
            else:
                accumulated.add_(batch_hvp, alpha=weight)
            del batch_loss, batch_hvp
        if accumulated is None:
            return torch.zeros_like(vector)
        return accumulated

    return hvp_fn


def build_full_normal_system(
    hvp_fn: TensorOperator,
    hg_full: torch.Tensor,
) -> tuple[torch.Tensor, TensorOperator]:
    def a_times(value: torch.Tensor) -> torch.Tensor:
        return hvp_fn(hvp_fn(value))

    return hg_full, a_times


def build_restricted_system_from_cached_hg(
    hvp_fn: TensorOperator,
    g_full: torch.Tensor,
    hg_full: torch.Tensor,
    index_list: torch.Tensor,
) -> tuple[torch.Tensor, TensorOperator]:
    full_dim = g_full.numel()
    rhs = project_subset(hg_full, index_list)
    full_buffer = torch.zeros(full_dim, device=g_full.device, dtype=g_full.dtype)

    def a_times(value: torch.Tensor) -> torch.Tensor:
        full_buffer.zero_()
        full_buffer.index_copy_(0, index_list, value)
        return project_subset(
            hvp_fn(hvp_fn(full_buffer)),
            index_list,
        )

    return rhs, a_times


def damping_candidates_for_solver(solver: str, args: argparse.Namespace) -> list[float]:
    if solver in {"p_lissa_full", "p_lissa_0p1"}:
        return [0.0]
    if solver == "schulz":
        return [0.0]
    override_name = f"{solver}_damping_candidates".replace("-", "_")
    override = getattr(args, override_name, None)
    return args.damping_candidates if override is None else override


def cgnr_inverse(
    a_times: TensorOperator,
    rhs: torch.Tensor,
    *,
    damping: float,
    normal_damping: float,
    tol: float,
    max_iter: int,
) -> dict[str, object]:
    def apply_shifted(value: torch.Tensor) -> torch.Tensor:
        output = a_times(value)
        if damping > 0:
            output = output + damping * value
        return output

    normal_rhs = apply_shifted(rhs)

    def normal_operator(value: torch.Tensor) -> torch.Tensor:
        return apply_shifted(apply_shifted(value))

    result = cg_inverse(
        a_times=normal_operator,
        rhs=normal_rhs,
        damping=normal_damping,
        tol=tol,
        max_iter=max_iter,
        return_details=True,
    )
    details = dict(result["details"])
    details["method"] = "cgnr_fallback"
    details["normal_damping"] = float(normal_damping)
    return {"solution": result["solution"], "details": details}


def run_solver_once(
    *,
    solver: str,
    base_a_times: TensorOperator,
    rhs: torch.Tensor,
    hvp_per_a_times: int,
    damping: float,
    args: argparse.Namespace,
    device: torch.device,
    model: torch.nn.Module,
    curvature_loader: DataLoader,
    curvature_inputs: torch.Tensor,
    curvature_targets: torch.Tensor,
    target_inputs: torch.Tensor,
    target_targets: torch.Tensor,
    criterion: torch.nn.Module,
) -> dict[str, object]:
    counted = CountedOperator(base_a_times, hvp_per_call=hvp_per_a_times)
    details: dict[str, object] = {}
    solution = torch.full_like(rhs, float("nan"))

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    idle_wait_sec = wait_for_idle_gpu(args, device)
    tracemalloc.start()
    synchronize(device)
    start = time.perf_counter()
    error = ""
    timed_out = False

    try:
        with solver_time_limit(args.time_limit_sec):
            if solver in {"p_lissa_full", "p_lissa_0p1"}:
                solution = p_lissa_inverse(
                    a_times=counted,
                    rhs=rhs,
                    damping=0.0,
                    mu=args.mu,
                    tol=args.tol,
                    max_iter=args.max_iter,
                    max_restarts=args.max_restarts,
                    power_iter_steps=args.power_iters,
                    keep_best_iterate=False,
                )
            elif solver == "lissa":
                result = lissa_inverse(
                    a_times=counted,
                    rhs=rhs,
                    damping=damping,
                    mu=args.lissa_mu,
                    tol=args.tol,
                    max_iter=args.max_iter,
                    power_iter_steps=args.power_iters,
                    return_details=True,
                )
                solution = result["solution"]
                details = result["details"]
            elif solver == "cg":
                try:
                    result = cg_inverse(
                        a_times=counted,
                        rhs=rhs,
                        damping=damping,
                        tol=args.tol,
                        max_iter=args.max_iter,
                        return_details=True,
                    )
                    result["details"]["method"] = "cg"
                except SolverTimeoutError:
                    raise
                except Exception:
                    if args.no_cg_cgnr_fallback:
                        raise
                    result = cgnr_inverse(
                        a_times=counted,
                        rhs=rhs,
                        damping=damping,
                        normal_damping=args.cg_cgnr_damping,
                        tol=args.tol,
                        max_iter=args.max_iter,
                    )
                solution = result["solution"]
                details = result["details"]
            elif solver == "schulz":
                result = hyperinf_inverse(
                    a_times=counted,
                    rhs=rhs,
                    beta_scale=args.schulz_beta_scale,
                    tol=-1.0,
                    max_iter=args.schulz_max_iter,
                    power_iter_steps=args.power_iters,
                    return_details=True,
                )
                solution = result["solution"]
                details = result["details"]
            elif solver == "lanczos":
                result = lanczos_inverse(
                    a_times=counted,
                    rhs=rhs,
                    rank=args.lanczos_rank,
                    damping=damping,
                    tol=args.tol,
                    max_iter=args.lanczos_max_iter,
                    return_details=True,
                )
                solution = result["solution"]
                details = result["details"]
            elif solver == "kfac":
                result = kfac_update(
                    model=model,
                    retained_inputs=curvature_loader,
                    retained_targets=None,
                    target_inputs=target_inputs,
                    target_targets=target_targets,
                    criterion=criterion,
                    damping=damping,
                    device=device,
                    return_details=True,
                )
                solution = result["update"]
                details = result["details"]
            elif solver == "ekfac":
                result = ekfac_update(
                    model=model,
                    retained_inputs=curvature_loader,
                    retained_targets=None,
                    target_inputs=target_inputs,
                    target_targets=target_targets,
                    criterion=criterion,
                    damping=damping,
                    device=device,
                    return_details=True,
                )
                solution = result["update"]
                details = result["details"]
            else:
                raise ValueError(f"Unknown solver: {solver}")
    except SolverTimeoutError as exc:
        timed_out = True
        error = f"{type(exc).__name__}: {exc}"
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001 - benchmark records failures.
        error = f"{type(exc).__name__}: {exc}"
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not timed_out:
        synchronize(device)
    elapsed = time.perf_counter() - start
    cuda_peak_mb, python_peak_mb = current_peak_memory_mb(device)
    tracemalloc.stop()

    eval_residual = math.nan
    solver_residual = math.nan
    update_norm = math.nan
    scale_alpha = 1.0
    eval_a_times_calls = 0
    finite_update = False if timed_out else bool(torch.isfinite(solution).all().item())

    if not error and finite_update:
        before_eval_calls = counted.calls
        try:
            calibrate_lissa = solver == "lissa" and not args.no_lissa_scale_calibration
            calibrate_kfac = (
                solver in {"kfac", "ekfac"} and not args.no_kfac_scale_calibration
            )
            if calibrate_lissa or calibrate_kfac:
                Ax = counted(solution)
                denom = torch.dot(Ax, Ax)
                numer = torch.dot(Ax, rhs)
                if (
                    torch.isfinite(denom)
                    and torch.isfinite(numer)
                    and denom.abs().item() > 1e-30
                ):
                    scale_alpha = (numer / denom).item()
                    solution = solution * scale_alpha
            eval_residual, solver_residual = relative_residual(
                counted, solution, rhs, damping=damping
            )
            update_norm = torch.linalg.norm(solution).item()
        except Exception as exc:  # noqa: BLE001
            error = f"residual {type(exc).__name__}: {exc}"
        eval_a_times_calls = counted.calls - before_eval_calls

    finite_residual = math.isfinite(eval_residual) and math.isfinite(solver_residual)
    finite_norm = math.isfinite(update_norm)
    success = (not error) and finite_update and finite_residual and finite_norm
    status = "timeout" if timed_out else ("ok" if success else "failed")
    if timed_out:
        success = False
    elif (
        args.time_limit_sec is not None
        and math.isfinite(elapsed)
        and elapsed > args.time_limit_sec
    ):
        success = False
        status = "timeout"
        if not error:
            error = f"time_limit_exceeded: {elapsed:.3f}s > {args.time_limit_sec:.3f}s"
    solve_a_times_calls = counted.calls - eval_a_times_calls
    return {
        "success": success,
        "status": status,
        "error": error,
        "damping": damping,
        "eval_residual": eval_residual,
        "solver_residual": solver_residual,
        "time_sec": elapsed,
        "idle_wait_sec": idle_wait_sec,
        "cuda_peak_mb": cuda_peak_mb,
        "python_peak_mb": python_peak_mb,
        "a_times_calls": solve_a_times_calls,
        "eval_a_times_calls": eval_a_times_calls,
        "hvp_per_a_times": hvp_per_a_times,
        "hvp_equiv_calls": solve_a_times_calls * hvp_per_a_times,
        "eval_hvp_equiv_calls": eval_a_times_calls * hvp_per_a_times,
        "update_norm": update_norm,
        "scale_alpha": scale_alpha,
        "method": details.get("method"),
        "iterations": details.get("iterations"),
        "converged": details.get("converged"),
        "details": details,
    }


def run_solver_with_damping_search(
    *,
    solver: str,
    system: str,
    base_a_times: TensorOperator,
    rhs: torch.Tensor,
    hvp_per_a_times: int,
    args: argparse.Namespace,
    device: torch.device,
    model: torch.nn.Module,
    curvature_loader: DataLoader,
    curvature_inputs: torch.Tensor,
    curvature_targets: torch.Tensor,
    target_inputs: torch.Tensor,
    target_targets: torch.Tensor,
    criterion: torch.nn.Module,
) -> dict[str, object]:
    candidates = damping_candidates_for_solver(solver, args)
    attempts = []
    best = None
    for damping in candidates:
        result = run_solver_once(
            solver=solver,
            base_a_times=base_a_times,
            rhs=rhs,
            hvp_per_a_times=hvp_per_a_times,
            damping=damping,
            args=args,
            device=device,
            model=model,
            curvature_loader=curvature_loader,
            curvature_inputs=curvature_inputs,
            curvature_targets=curvature_targets,
            target_inputs=target_inputs,
            target_targets=target_targets,
            criterion=criterion,
        )
        attempts.append(
            {
                "damping": damping,
                "success": result["success"],
                "status": result["status"],
                "eval_residual": result["eval_residual"],
                "solver_residual": result["solver_residual"],
                "error": result["error"],
            }
        )
        if result["success"]:
            best = result
            break
        if result["status"] == "timeout":
            best = result
            break

    if best is None:
        fallback_status = (
            "timeout"
            if any(attempt.get("status") == "timeout" for attempt in attempts)
            else "failed"
        )
        best = {
            "success": False,
            "status": fallback_status,
            "error": attempts[-1]["error"] if attempts else "no damping candidates",
            "damping": math.nan,
            "eval_residual": math.nan,
            "solver_residual": math.nan,
            "time_sec": math.nan,
            "idle_wait_sec": 0.0,
            "cuda_peak_mb": math.nan if device.type == "cuda" else None,
            "python_peak_mb": math.nan,
            "a_times_calls": 0,
            "eval_a_times_calls": 0,
            "hvp_per_a_times": hvp_per_a_times,
            "hvp_equiv_calls": 0,
            "eval_hvp_equiv_calls": 0,
            "update_norm": math.nan,
            "scale_alpha": math.nan,
            "method": None,
            "iterations": None,
            "converged": None,
            "details": {},
        }

    best = dict(best)
    best.update(
        {
            "solver": solver,
            "system": system,
            "rhs_dim": int(rhs.numel()),
            "rhs_norm": torch.linalg.norm(rhs).item(),
            "damping_candidates": candidates,
            "damping_attempts": attempts,
        }
    )
    return best


def skipped_row(
    *,
    solver: str,
    reason: str,
    system: str,
    hvp_per_a_times: int,
    size: str,
    path: Path,
    checkpoint: dict,
    target_params: int,
    actual_params: int,
    small_selected_params: int,
    p_lissa_small_ratio: float,
    classical_rhs_norm: float,
) -> dict[str, object]:
    return {
        "size": size,
        "solver": solver,
        "success": False,
        "status": "skipped",
        "error": reason,
        "damping": math.nan,
        "eval_residual": math.nan,
        "solver_residual": math.nan,
        "time_sec": 0.0,
        "idle_wait_sec": 0.0,
        "cuda_peak_mb": math.nan,
        "python_peak_mb": math.nan,
        "a_times_calls": 0,
        "eval_a_times_calls": 0,
        "hvp_per_a_times": hvp_per_a_times,
        "hvp_equiv_calls": 0,
        "eval_hvp_equiv_calls": 0,
        "update_norm": math.nan,
        "scale_alpha": math.nan,
        "method": None,
        "iterations": None,
        "converged": None,
        "system": system,
        "rhs_dim": 0,
        "rhs_norm": math.nan,
        "damping_candidates": [],
        "damping_attempts": [],
        "checkpoint": str(path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_acc": checkpoint.get("val_acc"),
        "target_params": target_params,
        "actual_params": actual_params,
        "base_width": int(checkpoint.get("base_width", 0)),
        "image_size": int(checkpoint.get("image_size", 0)),
        "p_lissa_small_ratio": p_lissa_small_ratio,
        "p_lissa_small_selected_params": small_selected_params,
        "classical_rhs_norm": classical_rhs_norm,
    }


def checkpoint_path(args: argparse.Namespace, size: str) -> Path:
    return args.checkpoint_dir / f"cifar10_scaled_resnet18_{size}.pth"


def result_row_key(row: dict) -> tuple[str, str, str]:
    return (
        str(row.get("seed", "")),
        str(row.get("size", "")),
        str(row.get("solver", "")),
    )


def output_prefix(args: argparse.Namespace) -> Path:
    return args.output_dir / "cifar10_solver_accuracy_scaling"


def seed_output_prefix(args: argparse.Namespace, seed: int) -> Path:
    return (
        args.output_dir / "cifar10" / f"seed_{seed}" / "cifar10_solver_accuracy_scaling"
    )


def load_existing_rows(path_prefix: Path) -> list[dict]:
    json_path = path_prefix.with_suffix(".json")
    csv_path = path_prefix.with_suffix(".csv")
    if json_path.is_file():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        return list(payload.get("results", []))
    if csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    return []


def merge_rows(*row_groups: list[dict]) -> list[dict]:
    merged: dict[tuple[str, str, str], dict] = {}
    for group in row_groups:
        for row in group:
            merged[result_row_key(row)] = row
    return list(merged.values())


def row_sort_key(row: dict) -> tuple[int, int, int, str, str, str]:
    try:
        seed_index = int(str(row.get("seed", 0)))
    except ValueError:
        seed_index = 0
    size_value = str(row.get("size", ""))
    solver_value = str(row.get("solver", ""))
    try:
        size_index = CIFAR10_SIZES.index(size_value)
    except ValueError:
        size_index = len(CIFAR10_SIZES)
    try:
        solver_index = SOLVERS.index(solver_value)
    except ValueError:
        solver_index = len(SOLVERS)
    return (
        seed_index,
        size_index,
        solver_index,
        size_value,
        solver_value,
        str(row.get("status", "")),
    )


def write_results(
    args: argparse.Namespace,
    device: torch.device,
    rows: list[dict],
    *,
    path_prefix: Path | None = None,
    seeds: list[int] | None = None,
    print_table: bool = True,
) -> None:
    if path_prefix is None:
        path_prefix = output_prefix(args)
    if seeds is None:
        seeds = resolve_seeds(args)
    rows = sorted(merge_rows(load_existing_rows(path_prefix), rows), key=row_sort_key)

    payload = {
        "config": {
            "checkpoint_dir": str(args.checkpoint_dir),
            "sizes": args.sizes,
            "solvers": args.solvers,
            "force": args.force,
            "seeds": seeds,
            "device": str(device),
            "dtype": "float32",
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "target_label": args.target_label,
            "target_batch_size": args.target_batch_size,
            "p_lissa_small_ratio": args.p_lissa_small_ratio,
            "tol": args.tol,
            "max_iter": args.max_iter,
            "damping_candidates": args.damping_candidates,
            "p_lissa_damping": 0.0,
            "cg_cgnr_fallback": not args.no_cg_cgnr_fallback,
            "cg_cgnr_damping": args.cg_cgnr_damping,
            "lissa_scale_calibration": not args.no_lissa_scale_calibration,
            "kfac_scale_calibration": not args.no_kfac_scale_calibration,
            "hvp_num_batches": args.hvp_num_batches,
            "time_limit_sec": args.time_limit_sec,
            "keep_running_after_timeout": args.keep_running_after_timeout,
            "wait_for_idle_gpu": args.wait_for_idle_gpu,
            "idle_gpu_util_threshold": args.idle_gpu_util_threshold,
            "idle_external_memory_mb": args.idle_external_memory_mb,
            "idle_check_interval_sec": args.idle_check_interval_sec,
            "idle_timeout_sec": args.idle_timeout_sec,
        },
        "results": rows,
    }

    path_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = path_prefix.with_suffix(".json")
    csv_path = path_prefix.with_suffix(".csv")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fieldnames = [
        "seed",
        "size",
        "solver",
        "status",
        "success",
        "damping",
        "eval_residual",
        "solver_residual",
        "time_sec",
        "idle_wait_sec",
        "cuda_peak_mb",
        "python_peak_mb",
        "a_times_calls",
        "eval_a_times_calls",
        "hvp_per_a_times",
        "hvp_equiv_calls",
        "eval_hvp_equiv_calls",
        "update_norm",
        "scale_alpha",
        "method",
        "iterations",
        "converged",
        "system",
        "rhs_dim",
        "rhs_norm",
        "target_params",
        "actual_params",
        "base_width",
        "image_size",
        "checkpoint_epoch",
        "checkpoint_val_acc",
        "p_lissa_small_ratio",
        "p_lissa_small_selected_params",
        "classical_rhs_norm",
        "checkpoint",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")
    if print_table:
        print_summary(rows)


def row_memory_mb(row: dict) -> float:
    cuda_peak_mb = row.get("cuda_peak_mb")
    if cuda_peak_mb is not None and math.isfinite(float(cuda_peak_mb)):
        return float(cuda_peak_mb)
    python_peak_mb = row.get("python_peak_mb")
    if python_peak_mb is not None and math.isfinite(float(python_peak_mb)):
        return float(python_peak_mb)
    return math.nan


def print_summary(rows: list[dict]) -> None:
    print("\nSummary")
    has_seed = any("seed" in row for row in rows)
    seed_header = f"{'seed':>6s}  " if has_seed else ""
    print(
        f"{seed_header}{'size':>6s}  {'solver':12s}  {'status':8s}  "
        f"{'residual (%)':>12s}  {'memory (MB)':>12s}  {'elapsed time (s)':>16s}"
    )
    for row in rows:
        residual_pct = float(row.get("eval_residual", math.nan)) * 100.0
        memory_mb = row_memory_mb(row)
        elapsed_s = float(row.get("time_sec", math.nan))
        seed_value = f"{str(row.get('seed', '')):>6s}  " if has_seed else ""
        print(
            f"{seed_value}{str(row.get('size', '')):>6s}  "
            f"{str(row.get('solver', '')):12s}  "
            f"{str(row.get('status', '')):8s}  "
            f"{residual_pct:12.4f}  "
            f"{memory_mb:12.2f}  "
            f"{elapsed_s:16.3f}"
        )


def run_seed(
    args: argparse.Namespace,
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    rows: list[dict],
    existing_keys: set[tuple[str, str, str]],
) -> bool:
    seed_args = copy(args)
    seed_args.seed = seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    criterion = torch.nn.CrossEntropyLoss()
    (
        curvature_loader,
        curvature_inputs,
        curvature_targets,
        target_inputs,
        target_targets,
    ) = load_batches(seed_args, device, dtype)

    print(f"\nseed={seed}")
    for size in seed_args.sizes:
        pending_solvers = [
            solver
            for solver in seed_args.solvers
            if seed_args.force or (str(seed), str(size), solver) not in existing_keys
        ]
        if not pending_solvers:
            print(f"\nsize={size} already up to date; skipping")
            continue

        path = checkpoint_path(seed_args, size)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        model, checkpoint = load_model(path, device, dtype)
        actual_params = int(checkpoint.get("actual_params", count_parameters(model)))
        target_params = int(checkpoint.get("target_params", 0))

        target_outputs = model(target_inputs)
        target_loss = criterion(target_outputs, target_targets)
        g_full = compute_gradient(model, target_loss, retain_graph=True)
        curvature_hvp = make_batched_hvp_fn(
            model=model,
            inputs=curvature_inputs,
            targets=curvature_targets,
            criterion=criterion,
            device=device,
            num_batches=seed_args.hvp_num_batches,
        )
        hg_full: torch.Tensor | None = None
        small_indices: torch.Tensor | None = None

        def get_hg_full() -> torch.Tensor:
            nonlocal hg_full
            if hg_full is None:
                hg_full = curvature_hvp(g_full)
            return hg_full

        def get_small_indices() -> torch.Tensor:
            nonlocal small_indices
            if small_indices is None:
                small_indices = select_indices(
                    model=model,
                    target_inputs=target_inputs,
                    target_targets=target_targets,
                    criterion=criterion,
                    ratio=seed_args.p_lissa_small_ratio,
                    device=device,
                )
            return small_indices

        def classical_h_times(value: torch.Tensor) -> torch.Tensor:
            return curvature_hvp(value)

        classical_rhs_norm = torch.linalg.norm(g_full).item()
        small_selected_params = (
            int(get_small_indices().numel())
            if "p_lissa_0p1" in seed_args.solvers
            else 0
        )
        print(f"\nsize={size}")

        for solver in seed_args.solvers:
            key = (str(seed), str(size), solver)
            if not seed_args.force and key in existing_keys:
                print(f"  {solver:12s} reused existing result")
                continue
            if solver == "p_lissa_full":
                system = "restricted_normal_full"
                hvp_per_a_times = 2
            elif solver == "p_lissa_0p1":
                system = "restricted_normal_0p1"
                hvp_per_a_times = 2
            else:
                system = "classical_hessian"
                hvp_per_a_times = 1

            if solver == "p_lissa_full":
                rhs, a_times = build_full_normal_system(
                    hvp_fn=curvature_hvp,
                    hg_full=get_hg_full(),
                )
            elif solver == "p_lissa_0p1":
                rhs, a_times = build_restricted_system_from_cached_hg(
                    hvp_fn=curvature_hvp,
                    g_full=g_full,
                    hg_full=get_hg_full(),
                    index_list=get_small_indices(),
                )
            else:
                rhs = g_full
                a_times = classical_h_times

            row = run_solver_with_damping_search(
                solver=solver,
                system=system,
                base_a_times=a_times,
                rhs=rhs,
                hvp_per_a_times=hvp_per_a_times,
                args=seed_args,
                device=device,
                model=model,
                curvature_loader=curvature_loader,
                curvature_inputs=curvature_inputs,
                curvature_targets=curvature_targets,
                target_inputs=target_inputs,
                target_targets=target_targets,
                criterion=criterion,
            )
            row.update(
                {
                    "seed": seed,
                    "size": size,
                    "checkpoint": str(path),
                    "checkpoint_epoch": checkpoint.get("epoch"),
                    "checkpoint_val_acc": checkpoint.get("val_acc"),
                    "target_params": target_params,
                    "actual_params": actual_params,
                    "base_width": int(checkpoint.get("base_width", 0)),
                    "image_size": int(checkpoint.get("image_size", 0)),
                    "p_lissa_small_ratio": seed_args.p_lissa_small_ratio,
                    "p_lissa_small_selected_params": small_selected_params,
                    "classical_rhs_norm": classical_rhs_norm,
                }
            )
            rows.append(row)
            existing_keys.add(key)
            mem_cuda = (
                "None"
                if row["cuda_peak_mb"] is None
                else f"{float(row['cuda_peak_mb']):.2f}"
            )
            print(
                f"  {solver:12s} eval_res={row['eval_residual']:.3e} "
                f"time={row['time_sec']:.3f}s mem_cuda={mem_cuda}MB "
                f"HVP~={row['hvp_equiv_calls']}"
            )
            if row["error"]:
                print(f"    error: {row['error']}")
            if row["status"] == "timeout" and seed_args.stop_after_timeout:
                print("    stopping: solver exceeded time limit")
                return False

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return True


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.float32
    rows = sorted(load_existing_rows(output_prefix(args)), key=row_sort_key)
    existing_keys = {result_row_key(row) for row in rows}

    for seed in resolve_seeds(args):
        seed_prefix = seed_output_prefix(args, seed)
        seed_existing_rows = load_existing_rows(seed_prefix)
        for row in seed_existing_rows:
            key = result_row_key(row)
            if key not in existing_keys:
                rows.append(row)
                existing_keys.add(key)
        seed_start = len(rows)
        should_continue = run_seed(
            args,
            seed=seed,
            device=device,
            dtype=dtype,
            rows=rows,
            existing_keys=existing_keys,
        )
        seed_rows = rows[seed_start:]
        if seed_rows:
            write_results(
                args,
                device,
                seed_rows,
                path_prefix=seed_output_prefix(args, seed),
                seeds=[seed],
                print_table=False,
            )
        if not should_continue:
            break

    write_results(args, device, rows)


if __name__ == "__main__":
    main()
