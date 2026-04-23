#!/usr/bin/env python3
"""Detailed profiler for HVP-heavy influence solvers on CIFAR-10 ResNet18."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn

SCRIPT_ROOT = Path(__file__).resolve().parent
SEARCH_ROOT = SCRIPT_ROOT.parents[1] / "search"
PROJECT_ROOT = SCRIPT_ROOT.parents[2]

for path in (SEARCH_ROOT, PROJECT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from _mnist_unlearning_common import (  # noqa: E402
    collect_retained_examples,
    collect_target_examples,
    load_checkpoint,
    set_seed,
)
from gif.data.huggingface import create_hf_data_bundle  # noqa: E402
from gif.influence import embed_subset, hvp  # noqa: E402
from gif.influence.common import compute_gradient  # noqa: E402
from gif.influence.restricted import build_restricted_system_from_hvp  # noqa: E402
from gif.models import ResNet18  # noqa: E402
from gif.selection import HighestKGradients  # noqa: E402
from gif.solvers.iterative import _estimate_lmax_power  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile HVP-heavy solver bottlenecks on CIFAR-10 ResNet18."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "hf_cifar10_resnet18.pth",
    )
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--dataset-id", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-target-batches", type=int, default=2)
    parser.add_argument(
        "--method",
        choices=["classical_if", "second_order_if", "freezing"],
        default="freezing",
    )
    parser.add_argument("--param-ratio", type=float, default=0.05)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--mu", type=float, default=3.0)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--power-iters", type=int, default=2)
    parser.add_argument("--lissa-damping", type=float, default=1e-2)
    parser.add_argument("--lissa-mu-scale", type=float, default=2.0)
    parser.add_argument("--lissa-max-restarts", type=int, default=12)
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass
class BatchRecord:
    examples: int
    elapsed_s: float


@dataclass
class HVPStats:
    call_count: int = 0
    total_elapsed_s: float = 0.0
    batch_count: int = 0
    batch_elapsed_s: float = 0.0
    batch_records: list[BatchRecord] = field(default_factory=list)

    def summary(self) -> dict[str, float | int]:
        avg_call = self.total_elapsed_s / self.call_count if self.call_count else 0.0
        avg_batch = self.batch_elapsed_s / self.batch_count if self.batch_count else 0.0
        return {
            "call_count": self.call_count,
            "total_elapsed_s": self.total_elapsed_s,
            "avg_call_s": avg_call,
            "batch_count": self.batch_count,
            "avg_batch_s": avg_batch,
        }


@dataclass
class SolveSummary:
    method: str
    selector_elapsed_s: float
    target_grad_elapsed_s: float
    initial_rhs_elapsed_s: float
    power_iteration_elapsed_s: float
    operator_call_count: int
    operator_elapsed_s: float
    hvp_stats: dict[str, float | int]
    iteration_records: list[dict[str, float | int]]
    status: str


def build_model() -> nn.Module:
    return ResNet18(in_channels=3)


def build_highest_gradient_selector(
    model: nn.Module,
    param_ratio: float,
    sampled_inputs: torch.Tensor,
    sampled_targets: torch.Tensor,
    criterion: nn.Module,
    device: torch.device,
) -> HighestKGradients:
    selector = HighestKGradients(model, param_ratio)
    selector.register_hooks()
    model.zero_grad(set_to_none=True)
    loss = criterion(model(sampled_inputs.to(device)), sampled_targets.to(device))
    loss.backward()
    selector.remove_hooks()
    model.zero_grad(set_to_none=True)
    return selector


def sample_target_batches(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
    num_batches: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_samples = min(batch_size * num_batches, len(inputs))
    permutation = torch.randperm(len(targets))[:num_samples]
    return inputs[permutation], targets[permutation]


def make_profiled_hvp_fn(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    device: torch.device,
    batch_size: int,
) -> tuple[callable, HVPStats]:
    stats = HVPStats()
    total_examples = len(targets)

    def hvp_fn(vector: torch.Tensor) -> torch.Tensor:
        accumulated = None
        stats.call_count += 1
        call_start = time.perf_counter()
        for start in range(0, total_examples, batch_size):
            batch_inputs = inputs[start : start + batch_size].to(device, non_blocking=True)
            batch_targets = targets[start : start + batch_size].to(
                device, non_blocking=True
            )
            synchronize(device)
            batch_start = time.perf_counter()
            batch_loss = criterion(model(batch_inputs), batch_targets)
            batch_hvp = hvp(model, batch_loss, vector)
            weight = batch_targets.size(0) / total_examples
            if accumulated is None:
                accumulated = batch_hvp * weight
            else:
                accumulated.add_(batch_hvp, alpha=weight)
            synchronize(device)
            batch_elapsed = time.perf_counter() - batch_start
            stats.batch_count += 1
            stats.batch_elapsed_s += batch_elapsed
            stats.batch_records.append(
                BatchRecord(examples=int(batch_targets.size(0)), elapsed_s=batch_elapsed)
            )
            del batch_loss, batch_hvp, batch_inputs, batch_targets
        synchronize(device)
        stats.total_elapsed_s += time.perf_counter() - call_start
        return accumulated if accumulated is not None else torch.zeros_like(vector)

    return hvp_fn, stats


def profile_operator(a_times, rhs: torch.Tensor, device: torch.device):
    operator_stats = {"count": 0, "elapsed_s": 0.0}

    def wrapped(value: torch.Tensor) -> torch.Tensor:
        synchronize(device)
        start = time.perf_counter()
        output = a_times(value)
        synchronize(device)
        operator_stats["count"] += 1
        operator_stats["elapsed_s"] += time.perf_counter() - start
        return output

    synchronize(device)
    power_start = time.perf_counter()
    lam_max_hat = _estimate_lmax_power(
        A_times=wrapped,
        dim=rhs.numel(),
        device=rhs.device,
        dtype=rhs.dtype,
        num_iter=2,
    )
    synchronize(device)
    power_elapsed = time.perf_counter() - power_start
    return wrapped, operator_stats, power_elapsed, lam_max_hat


def profiled_lissa(
    a_times,
    rhs: torch.Tensor,
    *,
    device: torch.device,
    damping: float,
    mu: float,
    tol: float,
    max_iter: int,
    power_iters: int,
) -> tuple[dict[str, object], dict[str, float | int], float]:
    operator_stats = {"count": 0, "elapsed_s": 0.0}

    def wrapped(value: torch.Tensor) -> torch.Tensor:
        synchronize(device)
        start = time.perf_counter()
        out = a_times(value)
        if damping > 0:
            out = out + damping * value
        synchronize(device)
        operator_stats["count"] += 1
        operator_stats["elapsed_s"] += time.perf_counter() - start
        return out

    power_elapsed = 0.0
    lam_max_hat = None
    if power_iters > 0:
        synchronize(device)
        power_start = time.perf_counter()
        lam_max_hat = _estimate_lmax_power(
            A_times=wrapped,
            dim=rhs.numel(),
            device=rhs.device,
            dtype=rhs.dtype,
            num_iter=power_iters,
        )
        synchronize(device)
        power_elapsed = time.perf_counter() - power_start
        if lam_max_hat is not None:
            mu = min(mu, 0.9 / max(lam_max_hat, 1e-12))

    x = mu * rhs.clone()
    rhs_norm = torch.linalg.norm(rhs).item() + 1e-12
    iteration_records = []
    status = "max_iter"
    for iteration in range(max_iter):
        synchronize(device)
        iter_start = time.perf_counter()
        Ax = wrapped(x)
        residual = rhs - Ax
        step = mu * residual
        x_next = x + step
        synchronize(device)
        iter_elapsed = time.perf_counter() - iter_start
        rel_residual = torch.linalg.norm(residual).item() / rhs_norm
        iteration_records.append(
            {
                "iteration": iteration + 1,
                "elapsed_s": iter_elapsed,
                "rel_residual": rel_residual,
                "x_norm": torch.linalg.norm(x).item(),
                "step_norm": torch.linalg.norm(step).item(),
            }
        )
        x = x_next
        if rel_residual < tol:
            status = "converged"
            break
        if not torch.isfinite(x).all():
            status = "non_finite"
            break
    return (
        {
            "solution": x,
            "iteration_records": iteration_records,
            "status": status,
            "lam_max_hat": lam_max_hat,
        },
        operator_stats,
        power_elapsed,
    )


def profile_freezing(
    model: nn.Module,
    target_loss: torch.Tensor,
    index_list,
    hvp_fn,
    *,
    device: torch.device,
    tol: float,
    max_iter: int,
) -> tuple[list[dict[str, float | int]], str]:
    grad = compute_gradient(model, target_loss)
    zero_mask = torch.ones(len(grad), dtype=torch.bool, device=grad.device)
    zero_mask[index_list] = False
    grad[zero_mask] = 0

    synchronize(device)
    initial_start = time.perf_counter()
    initial = hvp_fn(grad)
    synchronize(device)
    initial_elapsed = time.perf_counter() - initial_start
    initial[zero_mask] = 0

    diff_tol = tol * len(index_list) ** 0.5
    diff = diff_tol + 0.1
    diff_old = 1e10
    current = initial
    records = [{"stage": "initial_hvp", "elapsed_s": initial_elapsed}]
    status = "max_iter"

    for iteration in range(max_iter):
        previous = current
        synchronize(device)
        first_start = time.perf_counter()
        first_hvp = hvp_fn(previous)
        synchronize(device)
        first_elapsed = time.perf_counter() - first_start
        first_hvp[zero_mask] = 0

        synchronize(device)
        second_start = time.perf_counter()
        second_hvp = hvp_fn(first_hvp)
        synchronize(device)
        second_elapsed = time.perf_counter() - second_start
        second_hvp[zero_mask] = 0

        current = initial + previous - second_hvp
        diff = torch.norm(current - previous).item()
        records.append(
            {
                "iteration": iteration + 1,
                "first_hvp_elapsed_s": first_elapsed,
                "second_hvp_elapsed_s": second_elapsed,
                "diff": diff,
            }
        )
        if diff <= diff_tol:
            status = "converged"
            break
        if iteration % 2 == 0 and diff > diff_old:
            status = "diverged"
            break
        diff_old = diff

    return records, status


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    criterion = nn.CrossEntropyLoss()

    bundle = create_hf_data_bundle(
        "cifar10",
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        validation=True,
        flatten=False,
        seed=args.seed,
        dataset_id=args.dataset_id,
    )
    train_loader = bundle.train_loader

    model = build_model().to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    all_target_inputs, all_target_targets = collect_target_examples(
        train_loader, args.target_label
    )
    retained_inputs, retained_targets = collect_retained_examples(
        train_loader, args.target_label, 1
    )
    sampled_inputs, sampled_targets = sample_target_batches(
        all_target_inputs,
        all_target_targets,
        args.batch_size,
        args.num_target_batches,
    )

    selector_elapsed = 0.0
    if args.method == "freezing":
        synchronize(device)
        start = time.perf_counter()
        selector = build_highest_gradient_selector(
            model,
            args.param_ratio,
            sampled_inputs,
            sampled_targets,
            criterion,
            device,
        )
        synchronize(device)
        selector_elapsed = time.perf_counter() - start
        index_list = selector.get_parameters()
    else:
        index_list = list(range(sum(p.numel() for p in model.parameters())))

    target_scaling = all_target_targets.size(0) / (
        len(train_loader.dataset) - all_target_targets.size(0)
    )

    model.zero_grad(set_to_none=True)
    target_loss = (
        criterion(model(sampled_inputs.to(device)), sampled_targets.to(device))
        * target_scaling
    )

    hvp_fn, hvp_stats = make_profiled_hvp_fn(
        model=model,
        inputs=retained_inputs,
        targets=retained_targets,
        criterion=criterion,
        device=device,
        batch_size=args.batch_size,
    )

    if args.method == "freezing":
        target_grad_elapsed = 0.0
        initial_rhs_elapsed = 0.0
        records, status = profile_freezing(
            model,
            target_loss,
            index_list,
            hvp_fn,
            device=device,
            tol=args.tol,
            max_iter=args.max_iter,
        )
        summary = SolveSummary(
            method=args.method,
            selector_elapsed_s=selector_elapsed,
            target_grad_elapsed_s=target_grad_elapsed,
            initial_rhs_elapsed_s=initial_rhs_elapsed,
            power_iteration_elapsed_s=0.0,
            operator_call_count=0,
            operator_elapsed_s=0.0,
            hvp_stats=hvp_stats.summary(),
            iteration_records=records,
            status=status,
        )
        print(json.dumps(asdict(summary), indent=2))
        return

    synchronize(device)
    grad_start = time.perf_counter()
    g_full = compute_gradient(
        model, target_loss, retain_graph=(args.method == "second_order_if")
    )
    synchronize(device)
    target_grad_elapsed = time.perf_counter() - grad_start

    synchronize(device)
    rhs_start = time.perf_counter()
    rhs, a_times = build_restricted_system_from_hvp(hvp_fn, g_full, index_list)
    synchronize(device)
    initial_rhs_elapsed = time.perf_counter() - rhs_start

    if args.method == "classical_if":
        solved, operator_stats, power_elapsed = profiled_lissa(
            a_times,
            rhs,
            device=device,
            damping=args.lissa_damping,
            mu=args.mu * args.lissa_mu_scale,
            tol=args.tol,
            max_iter=args.max_iter,
            power_iters=args.power_iters,
        )
        summary = SolveSummary(
            method=args.method,
            selector_elapsed_s=selector_elapsed,
            target_grad_elapsed_s=target_grad_elapsed,
            initial_rhs_elapsed_s=initial_rhs_elapsed,
            power_iteration_elapsed_s=power_elapsed,
            operator_call_count=operator_stats["count"],
            operator_elapsed_s=operator_stats["elapsed_s"],
            hvp_stats=hvp_stats.summary(),
            iteration_records=solved["iteration_records"],
            status=solved["status"],
        )
    elif args.method == "second_order_if":
        solved_first, operator_stats_first, power_elapsed_first = profiled_lissa(
            a_times,
            rhs,
            device=device,
            damping=args.lissa_damping,
            mu=args.mu * args.lissa_mu_scale,
            tol=args.tol,
            max_iter=args.max_iter,
            power_iters=args.power_iters,
        )
        ratio = all_target_targets.size(0) / len(train_loader.dataset)
        ratio_scale = ratio / (1.0 - ratio)
        index_tensor = torch.as_tensor(index_list, device=device, dtype=torch.long)
        full_dim = sum(parameter.numel() for parameter in model.parameters())
        first_order_full = embed_subset(
            solved_first["solution"] * ratio_scale, index_tensor, full_dim
        )
        synchronize(device)
        rhs2_start = time.perf_counter()
        second_rhs_full = hvp_fn(first_order_full) - hvp(model, target_loss, first_order_full)
        second_rhs, second_operator = build_restricted_system_from_hvp(
            hvp_fn, second_rhs_full, index_list
        )
        synchronize(device)
        second_rhs_elapsed = time.perf_counter() - rhs2_start
        solved_second, operator_stats_second, power_elapsed_second = profiled_lissa(
            second_operator,
            second_rhs,
            device=device,
            damping=args.lissa_damping,
            mu=args.mu * args.lissa_mu_scale,
            tol=args.tol,
            max_iter=args.max_iter,
            power_iters=args.power_iters,
        )
        summary = SolveSummary(
            method=args.method,
            selector_elapsed_s=selector_elapsed,
            target_grad_elapsed_s=target_grad_elapsed,
            initial_rhs_elapsed_s=initial_rhs_elapsed + second_rhs_elapsed,
            power_iteration_elapsed_s=power_elapsed_first + power_elapsed_second,
            operator_call_count=operator_stats_first["count"] + operator_stats_second["count"],
            operator_elapsed_s=operator_stats_first["elapsed_s"] + operator_stats_second["elapsed_s"],
            hvp_stats=hvp_stats.summary(),
            iteration_records=solved_first["iteration_records"] + solved_second["iteration_records"],
            status=f"first={solved_first['status']}, second={solved_second['status']}",
        )
    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
