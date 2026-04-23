#!/usr/bin/env python3
"""Shared influence benchmark runner for text transformer models."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn

SCRIPT_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_ROOT
SEARCH_ROOT = SCRIPT_ROOT.parents[1] / "search"
PROJECT_ROOT = SCRIPT_ROOT.parents[2]

for path in (SEARCH_ROOT, PROJECT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from _hf_text_unlearning_common import (  # noqa: E402
    build_loader,
    build_total_loss,
    collect_examples,
    collect_retained_examples,
    collect_target_examples,
    format_metrics,
    load_checkpoint,
    set_seed,
)

from gif.data.huggingface import create_hf_data_bundle  # noqa: E402
from gif.influence import (  # noqa: E402
    DataInfluence,
    EKFACInfluence,
    TracIn,
    embed_subset,
    hvp,
    project_subset,
)
from gif.influence.common import compute_gradient  # noqa: E402
from gif.influence.restricted import build_restricted_system_from_hvp  # noqa: E402
from gif.models import (  # noqa: E402
    TextTransformerClassifier,
    trainable_parameters_to_vector,
    vector_to_trainable_parameters,
)
from gif.selection import HighestKGradients  # noqa: E402
from gif.solvers import hyperinf_inverse, lissa_inverse, p_lissa_inverse  # noqa: E402

METHOD_SPECS = {
    "gif": {"label": "GIF", "uses_param_ratio": True},
    "classical_if": {"label": "Classic IF", "uses_param_ratio": False},
    "second_order_if": {"label": "Second-order IF", "uses_param_ratio": False},
    "tracin": {
        "label": "TracIn",
        "uses_param_ratio": False,
        "requires_trajectory": True,
    },
    "hypeinf": {"label": "HypeInf", "uses_param_ratio": False},
    "datainf": {"label": "DataInf", "uses_param_ratio": False},
    "freezing": {"label": "Freezing", "uses_param_ratio": True},
    "ekfac": {
        "label": "EKFAC",
        "uses_param_ratio": False,
        "requires_all_trainable": True,
    },
}

DEFAULT_METHODS = [
    "classical_if",
    "second_order_if",
    "tracin",
    "hypeinf",
    "datainf",
    "freezing",
    "ekfac",
    "gif",
]
DEFAULT_PARAM_RATIOS = [0.05]


class TrainableParameterSelector:
    def __init__(self, model: torch.nn.Module):
        self.model = model

    def get_parameters(self):
        return list(range(trainable_parameters_to_vector(self.model).numel()))

    def update_network(self, update: torch.Tensor) -> None:
        base = trainable_parameters_to_vector(self.model).detach()
        vector_to_trainable_parameters(
            base + update.to(base.device, dtype=base.dtype),
            self.model,
        )


def parse_args(
    default_checkpoint: Path, default_target_label: int
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark influence-based model edit schemes on a text transformer."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint)
    parser.add_argument("--trajectory-dir", type=Path, default=None)
    parser.add_argument("--tracin-max-checkpoints", type=int, default=10)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--dataset-id", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--target-label", type=int, default=default_target_label)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--num-target-batches", type=int, default=10)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(METHOD_SPECS.keys()),
        default=DEFAULT_METHODS,
    )
    parser.add_argument(
        "--param-ratios",
        nargs="+",
        type=float,
        default=DEFAULT_PARAM_RATIOS,
    )
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--mu", type=float, default=3.0)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--hypeinf-max-iter", type=int, default=5)
    parser.add_argument("--solver-power-iters", type=int, default=2)
    parser.add_argument("--edit-scale", type=float, default=0.01)
    parser.add_argument("--max-update-steps", type=int, default=200)
    parser.add_argument("--gif-max-self-acc-for-selection", type=float, default=1.5)
    parser.add_argument("--hyperinf-beta-scale", type=float, default=0.9)
    parser.add_argument("--datainf-damping", type=float, default=1e-6)
    parser.add_argument("--ekfac-damping", type=float, default=1e-3)
    parser.add_argument("--lissa-damping", type=float, default=1e-2)
    parser.add_argument("--lissa-mu-scale", type=float, default=2.0)
    parser.add_argument("--lissa-max-restarts", type=int, default=12)
    parser.add_argument("--max-text-length", type=int, default=128)
    parser.add_argument("--max-vocab-size", type=int, default=30000)
    parser.add_argument("--min-token-freq", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--text-num-layers", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=512)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise instead of recording skipped entries for incompatible methods.",
    )
    return parser.parse_args()


def build_model(bundle, args: argparse.Namespace) -> nn.Module:
    return TextTransformerClassifier(
        vocab_size=min(bundle.vocabulary.size, args.max_vocab_size),
        max_len=args.max_text_length,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.text_num_layers,
        dim_feedforward=args.dim_feedforward,
        num_classes=bundle.num_classes,
    )


def build_retrained_checkpoint_path(args: argparse.Namespace) -> Path:
    checkpoint_path = args.checkpoint
    return checkpoint_path.with_name(
        f"{checkpoint_path.stem}_without_{args.target_label}{checkpoint_path.suffix}"
    )


def build_trajectory_path(args: argparse.Namespace) -> Path | None:
    if args.trajectory_dir is not None:
        return args.trajectory_dir
    candidate = args.checkpoint.parent / args.checkpoint.stem
    if candidate.is_dir():
        return candidate
    return None


def sample_target_batches(
    input_ids: torch.Tensor,
    attention_masks: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
    num_batches: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_samples = min(batch_size * num_batches, len(targets))
    permutation = torch.randperm(len(targets))[:num_samples]
    return (
        input_ids[permutation],
        attention_masks[permutation],
        targets[permutation],
    )


def sample_hvp_batch(
    input_ids: torch.Tensor,
    attention_masks: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(targets) == 0:
        raise RuntimeError("Cannot build HVP batch from an empty retained set.")
    if len(targets) <= batch_size:
        return input_ids, attention_masks, targets
    permutation = torch.randperm(len(targets))[:batch_size]
    return input_ids[permutation], attention_masks[permutation], targets[permutation]


def collect_eval_splits(dataloader, target_label: int):
    input_ids, attention_masks, labels = collect_examples(dataloader)
    target_mask = labels == target_label
    retain_mask = ~target_mask
    if not torch.any(target_mask) or not torch.any(retain_mask):
        raise RuntimeError("Could not build text evaluation splits.")
    return (
        input_ids[target_mask],
        attention_masks[target_mask],
        labels[target_mask],
        input_ids[retain_mask],
        attention_masks[retain_mask],
        labels[retain_mask],
    )


def evaluate_split_tensors(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_masks: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    device: torch.device,
    batch_size: int,
) -> tuple[float, float]:
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.inference_mode():
        for start in range(0, len(targets), batch_size):
            ids = input_ids[start : start + batch_size].to(device, non_blocking=True)
            masks = attention_masks[start : start + batch_size].to(
                device, non_blocking=True
            )
            batch_targets = targets[start : start + batch_size].to(
                device, non_blocking=True
            )
            outputs = model(ids, masks)
            loss = criterion(outputs, batch_targets)
            total_loss += loss.item() * batch_targets.size(0)
            total_correct += outputs.argmax(dim=1).eq(batch_targets).sum().item()
            total_examples += batch_targets.size(0)
    if total_examples == 0:
        raise RuntimeError("Split evaluation received zero examples.")
    return total_loss / total_examples, 100.0 * total_correct / total_examples


def evaluate_unlearning_cached(
    model: nn.Module,
    target_ids: torch.Tensor,
    target_masks: torch.Tensor,
    target_targets: torch.Tensor,
    retain_ids: torch.Tensor,
    retain_masks: torch.Tensor,
    retain_targets: torch.Tensor,
    criterion: nn.Module,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    self_loss, self_acc = evaluate_split_tensors(
        model,
        target_ids,
        target_masks,
        target_targets,
        criterion,
        device,
        batch_size,
    )
    retain_loss, retain_acc = evaluate_split_tensors(
        model,
        retain_ids,
        retain_masks,
        retain_targets,
        criterion,
        device,
        batch_size,
    )
    self_acc_norm = self_acc / 100.0
    retain_acc_norm = retain_acc / 100.0
    score = 0.0
    if not (self_acc == 100.0 and retain_acc == 0.0):
        score = (
            2.0
            * (1.0 - self_acc_norm)
            * retain_acc_norm
            / (1.0 - self_acc_norm + retain_acc_norm)
        )
    return {
        "self_loss": self_loss,
        "self_acc": self_acc,
        "retain_loss": retain_loss,
        "retain_acc": retain_acc,
        "score": score,
    }


def make_batched_hvp_fn(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_masks: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    device: torch.device,
    batch_size: int,
):
    loader = build_loader(input_ids, attention_masks, targets, batch_size)
    total_examples = len(targets)

    def hvp_fn(vector: torch.Tensor) -> torch.Tensor:
        accumulated = None
        for (batch_ids, batch_masks), batch_targets in loader:
            batch_ids = batch_ids.to(device, non_blocking=True)
            batch_masks = batch_masks.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            batch_loss = criterion(model(batch_ids, batch_masks), batch_targets)
            batch_hvp = hvp(model, batch_loss, vector)
            weight = batch_targets.size(0) / total_examples
            if accumulated is None:
                accumulated = batch_hvp * weight
            else:
                accumulated.add_(batch_hvp, alpha=weight)
            del batch_loss, batch_hvp
        return accumulated if accumulated is not None else torch.zeros_like(vector)

    return hvp_fn


def lissa_update_from_hvp_fn(
    model: nn.Module,
    target_loss: torch.Tensor,
    index_list,
    hvp_fn,
    *,
    damping: float,
    mu: float,
    tol: float,
    max_iter: int,
    max_restarts: int,
    power_iter_steps: int,
    retain_graph: bool = False,
):
    g_full = compute_gradient(model, target_loss, retain_graph=retain_graph)
    rhs, a_times = build_restricted_system_from_hvp(hvp_fn, g_full, index_list)
    current_damping = damping
    current_mu = mu
    residual_accept = max(tol * 10.0, 1e-3)
    best_solution = None

    for _ in range(5):
        solved = lissa_inverse(
            a_times=a_times,
            rhs=rhs,
            damping=current_damping,
            mu=current_mu,
            tol=tol,
            max_iter=max_iter,
            max_restarts=max_restarts,
            power_iter_steps=power_iter_steps,
            return_details=True,
            verbose=False,
        )
        best_solution = solved["solution"]
        details = solved["details"]
        residuals = details.get("residuals", [])
        final_residual = residuals[-1] if residuals else float("inf")
        if details.get("converged", False) or final_residual <= residual_accept:
            return best_solution
        current_damping *= 5.0
        current_mu *= 0.5

    return best_solution


def p_lissa_update_from_hvp_fn(
    model: nn.Module,
    target_loss: torch.Tensor,
    index_list,
    hvp_fn,
    *,
    mu: float,
    tol: float,
    max_iter: int,
    power_iter_steps: int,
):
    g_full = compute_gradient(model, target_loss, retain_graph=False)
    rhs, a_times = build_restricted_system_from_hvp(hvp_fn, g_full, index_list)
    return p_lissa_inverse(
        a_times=a_times,
        rhs=rhs,
        mu=mu,
        tol=tol,
        max_iter=max_iter,
        max_restarts=8,
        power_iter_steps=power_iter_steps,
        verbose=False,
    )


def hyperinf_update_from_hvp_fn(
    model: nn.Module,
    target_loss: torch.Tensor,
    index_list,
    hvp_fn,
    *,
    beta_scale: float,
    tol: float,
    max_iter: int,
    power_iter_steps: int,
):
    g_full = compute_gradient(model, target_loss, retain_graph=False)
    rhs, a_times = build_restricted_system_from_hvp(hvp_fn, g_full, index_list)
    return hyperinf_inverse(
        a_times=a_times,
        rhs=rhs,
        beta=None,
        beta_scale=beta_scale,
        tol=tol,
        max_iter=max_iter,
        power_iter_steps=power_iter_steps,
        verbose=False,
    )


def freezing_update_from_hvp_fn(
    model: nn.Module,
    target_loss: torch.Tensor,
    index_list,
    hvp_fn,
    *,
    tol: float,
    step: float,
    max_iter: int,
):
    normalizer = 1.0
    while True:
        inv_normalizer = 1.0 / normalizer

        def scaled_hvp_fn(vector: torch.Tensor) -> torch.Tensor:
            return hvp_fn(vector) * inv_normalizer

        grad = compute_gradient(model, target_loss / normalizer)
        zero_mask = torch.ones(len(grad), dtype=torch.bool, device=grad.device)
        zero_mask[index_list] = False
        grad[zero_mask] = 0
        initial = scaled_hvp_fn(grad)
        initial[zero_mask] = 0

        diff_tol = tol * len(index_list) ** 0.5
        diff = diff_tol + 0.1
        diff_old = 1e10
        current = initial
        count = 0
        while diff > diff_tol and count < max_iter:
            previous = current
            first_hvp = scaled_hvp_fn(previous)
            first_hvp[zero_mask] = 0
            second_hvp = scaled_hvp_fn(first_hvp)
            second_hvp[zero_mask] = 0
            current = initial + previous - second_hvp
            diff = torch.norm(current - previous)
            if count % 2 == 0:
                if diff > diff_old:
                    current = None
                    break
                diff_old = diff
            count += 1

        if current is not None:
            return current[index_list]
        normalizer += step


def all_parameters_trainable(model: nn.Module) -> bool:
    parameters = list(model.parameters())
    return bool(parameters) and all(parameter.requires_grad for parameter in parameters)


def compatibility_issue(
    method_name: str,
    model: nn.Module,
    trajectory_dir: Path | None,
) -> str | None:
    spec = METHOD_SPECS[method_name]
    if spec.get("requires_all_trainable", False) and not all_parameters_trainable(
        model
    ):
        return "requires all model parameters to be trainable"
    if method_name != "datainf" and not all_parameters_trainable(model):
        return "current implementation only supports fully trainable models for this method"
    if spec.get("requires_trajectory", False):
        if trajectory_dir is None:
            return "requires a trajectory directory"
        if not trajectory_dir.is_dir():
            return f"trajectory directory not found: {trajectory_dir}"
    return None


def build_highest_gradient_selector(
    model: nn.Module,
    param_ratio: float,
    sampled_ids: torch.Tensor,
    sampled_masks: torch.Tensor,
    sampled_targets: torch.Tensor,
    criterion: nn.Module,
    device: torch.device,
) -> HighestKGradients:
    selector = HighestKGradients(model, param_ratio)
    selector.register_hooks()
    model.zero_grad(set_to_none=True)
    target_loss = criterion(
        model(sampled_ids.to(device), sampled_masks.to(device)),
        sampled_targets.to(device),
    )
    target_loss.backward()
    selector.remove_hooks()
    model.zero_grad(set_to_none=True)
    return selector


def compute_method_update(
    method_name: str,
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    sampled_ids: torch.Tensor,
    sampled_masks: torch.Tensor,
    sampled_targets: torch.Tensor,
    all_target_count: int,
    retained_ids: torch.Tensor,
    retained_masks: torch.Tensor,
    retained_targets: torch.Tensor,
    param_ratio: float | None,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[object, torch.Tensor, int]:
    model.eval()
    if len(train_loader.dataset) <= all_target_count:
        raise RuntimeError("Target set size must be smaller than the training dataset.")
    target_scaling = all_target_count / (len(train_loader.dataset) - all_target_count)

    if method_name in {"gif", "freezing"}:
        if param_ratio is None:
            raise RuntimeError(f"{method_name} requires a parameter ratio.")
        selector = build_highest_gradient_selector(
            model=model,
            param_ratio=param_ratio,
            sampled_ids=sampled_ids,
            sampled_masks=sampled_masks,
            sampled_targets=sampled_targets,
            criterion=criterion,
            device=device,
        )
        index_list = selector.get_parameters()
    else:
        selector = TrainableParameterSelector(model)
        index_list = selector.get_parameters()

    if len(index_list) == 0:
        raise RuntimeError(f"{method_name} parameter selection returned an empty set.")

    index_tensor = torch.as_tensor(index_list, device=device, dtype=torch.long)
    full_dim = sum(parameter.numel() for parameter in model.parameters())

    model.zero_grad(set_to_none=True)
    target_loss = (
        criterion(
            model(sampled_ids.to(device), sampled_masks.to(device)),
            sampled_targets.to(device),
        )
        * target_scaling
    )
    hvp_ids, hvp_masks, hvp_targets = sample_hvp_batch(
        retained_ids, retained_masks, retained_targets, args.batch_size
    )
    retained_hvp_fn = make_batched_hvp_fn(
        model=model,
        input_ids=hvp_ids,
        attention_masks=hvp_masks,
        targets=hvp_targets,
        criterion=criterion,
        device=device,
        batch_size=args.batch_size,
    )

    if method_name == "gif":
        influence = p_lissa_update_from_hvp_fn(
            model,
            target_loss,
            index_list,
            retained_hvp_fn,
            mu=args.mu,
            tol=args.tol,
            max_iter=args.max_iter,
            power_iter_steps=args.solver_power_iters,
        )
    elif method_name == "classical_if":
        influence = lissa_update_from_hvp_fn(
            model,
            target_loss=target_loss,
            index_list=index_list,
            hvp_fn=retained_hvp_fn,
            damping=args.lissa_damping,
            mu=args.mu * args.lissa_mu_scale,
            tol=args.tol,
            max_iter=args.max_iter,
            max_restarts=args.lissa_max_restarts,
            power_iter_steps=args.solver_power_iters,
        )
    elif method_name == "second_order_if":
        ratio = all_target_count / len(train_loader.dataset)
        ratio_scale = ratio / (1.0 - ratio)
        first_order = lissa_update_from_hvp_fn(
            model,
            target_loss=target_loss,
            index_list=index_list,
            hvp_fn=retained_hvp_fn,
            damping=args.lissa_damping,
            mu=args.mu * args.lissa_mu_scale,
            tol=args.tol,
            max_iter=args.max_iter,
            max_restarts=args.lissa_max_restarts,
            power_iter_steps=args.solver_power_iters,
            retain_graph=True,
        )
        first_order = first_order * ratio_scale

        first_order_full = embed_subset(first_order, index_tensor, full_dim)
        second_order_rhs_full = retained_hvp_fn(first_order_full) - hvp(
            model, target_loss, first_order_full
        )
        second_rhs, second_operator = build_restricted_system_from_hvp(
            retained_hvp_fn, second_order_rhs_full, index_list
        )
        second_order = lissa_inverse(
            a_times=second_operator,
            rhs=second_rhs,
            damping=args.lissa_damping,
            mu=args.mu * args.lissa_mu_scale,
            tol=args.tol,
            max_iter=args.max_iter,
            max_restarts=args.lissa_max_restarts,
            power_iter_steps=args.solver_power_iters,
            verbose=False,
        )
        influence = first_order + second_order * ratio_scale
    elif method_name == "tracin":
        trajectory_dir = build_trajectory_path(args)
        if trajectory_dir is None:
            raise RuntimeError("TracIn requires a trajectory_dir.")
        influence = (
            TracIn().compute_update(
                model=model,
                trajectory_dir=trajectory_dir,
                target_inputs=(sampled_ids.to(device), sampled_masks.to(device)),
                target_targets=sampled_targets.to(device),
                criterion=criterion,
                device=device,
                max_checkpoints=args.tracin_max_checkpoints,
                index_list=index_list,
            )
            * target_scaling
        )
    elif method_name == "hypeinf":
        influence = hyperinf_update_from_hvp_fn(
            model,
            target_loss=target_loss,
            index_list=index_list,
            hvp_fn=retained_hvp_fn,
            beta_scale=args.hyperinf_beta_scale,
            tol=args.tol,
            max_iter=args.hypeinf_max_iter,
            power_iter_steps=args.solver_power_iters,
        )
    elif method_name == "datainf":
        total_loss = build_total_loss(
            model, retained_ids, retained_masks, retained_targets, criterion, device
        )
        influence = DataInfluence().compute(
            model=model,
            total_loss=total_loss,
            target_loss=target_loss,
            damping=args.datainf_damping,
        )
    elif method_name == "freezing":
        influence = freezing_update_from_hvp_fn(
            model,
            target_loss,
            index_list=index_list,
            hvp_fn=retained_hvp_fn,
            tol=args.tol,
            step=0.5,
            max_iter=args.max_iter,
        )
    elif method_name == "ekfac":
        influence = project_subset(
            EKFACInfluence().compute(
                model=model,
                retained_inputs=(retained_ids, retained_masks),
                retained_targets=retained_targets,
                target_inputs=(sampled_ids, sampled_masks),
                target_targets=sampled_targets,
                criterion=criterion,
                damping=args.ekfac_damping,
                device=device,
            ),
            index_list,
        )
    else:
        raise ValueError(f"Unsupported method: {method_name}")

    norm = torch.norm(influence)
    if torch.isnan(norm) or norm.item() == 0.0:
        raise RuntimeError(f"{method_name} update has zero or NaN norm.")

    return selector, influence / norm, int(len(index_list))


def build_skipped_result(
    method_name: str,
    param_ratio: float | None,
    trial_seed: int,
    before_metrics: dict[str, float],
    reason: str,
) -> dict[str, object]:
    return {
        "seed": trial_seed,
        "method": method_name,
        "method_label": METHOD_SPECS[method_name]["label"],
        "param_ratio": param_ratio,
        "status": "skipped",
        "reason": reason,
        "selected_params": 0,
        "reached_target": False,
        "target_step": None,
        "retain_acc": before_metrics["retain_acc"],
        "self_acc": before_metrics["self_acc"],
        "retain_loss": before_metrics["retain_loss"],
        "self_loss": before_metrics["self_loss"],
        "score": before_metrics["score"],
        "before_self_acc": before_metrics["self_acc"],
        "before_retain_acc": before_metrics["retain_acc"],
        "before_score": before_metrics["score"],
        "retain_acc_drop": 0.0,
        "self_acc_drop": 0.0,
    }


def format_trial_prefix(trial_seed: int, method_name: str) -> str:
    return f"[seed={trial_seed} | {method_name:<16}]"


def method_param_ratios(
    method_name: str, args: argparse.Namespace
) -> list[float | None]:
    if METHOD_SPECS[method_name]["uses_param_ratio"]:
        return list(args.param_ratios)
    return [None]


def run_single_method(
    method_name: str,
    param_ratio: float | None,
    args: argparse.Namespace,
    trial_seed: int,
    criterion: nn.Module,
    train_loader,
    sampled_ids: torch.Tensor,
    sampled_masks: torch.Tensor,
    sampled_targets: torch.Tensor,
    all_target_count: int,
    retained_ids: torch.Tensor,
    retained_masks: torch.Tensor,
    retained_targets: torch.Tensor,
    eval_target_ids: torch.Tensor,
    eval_target_masks: torch.Tensor,
    eval_target_targets: torch.Tensor,
    eval_retain_ids: torch.Tensor,
    eval_retain_masks: torch.Tensor,
    eval_retain_targets: torch.Tensor,
    base_state_dict: dict[str, torch.Tensor],
    model_factory,
    device: torch.device,
    trajectory_dir: Path | None,
) -> dict[str, object]:
    model = model_factory().to(device)
    model.load_state_dict(base_state_dict)
    model.eval()

    before_metrics = evaluate_unlearning_cached(
        model,
        eval_target_ids,
        eval_target_masks,
        eval_target_targets,
        eval_retain_ids,
        eval_retain_masks,
        eval_retain_targets,
        criterion,
        device,
        args.batch_size,
    )

    reason = compatibility_issue(method_name, model, trajectory_dir)
    if reason is not None:
        if args.strict:
            raise RuntimeError(reason)
        result = build_skipped_result(
            method_name=method_name,
            param_ratio=param_ratio,
            trial_seed=trial_seed,
            before_metrics=before_metrics,
            reason=reason,
        )
        print(f"{format_trial_prefix(trial_seed, method_name)} Skipped: {reason}")
        return result

    selector, normalized_update, selected_params = compute_method_update(
        method_name=method_name,
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        sampled_ids=sampled_ids,
        sampled_masks=sampled_masks,
        sampled_targets=sampled_targets,
        all_target_count=all_target_count,
        retained_ids=retained_ids,
        retained_masks=retained_masks,
        retained_targets=retained_targets,
        param_ratio=param_ratio,
        args=args,
        device=device,
    )

    best_metrics = before_metrics
    best_state_dict = deepcopy(model.state_dict())
    best_step = 0
    target_step = None

    selected_metrics = before_metrics
    for step_index in range(1, args.max_update_steps + 1):
        selector.update_network(normalized_update * args.edit_scale)
        current_metrics = evaluate_unlearning_cached(
            model,
            eval_target_ids,
            eval_target_masks,
            eval_target_targets,
            eval_retain_ids,
            eval_retain_masks,
            eval_retain_targets,
            criterion,
            device,
            args.batch_size,
        )
        if target_step is None and current_metrics["self_acc"] <= 0.0:
            target_step = step_index

        eligible = True
        if method_name == "gif":
            eligible = current_metrics["self_acc"] < args.gif_max_self_acc_for_selection
        if eligible and current_metrics["score"] >= best_metrics["score"]:
            best_metrics = current_metrics
            best_state_dict = deepcopy(model.state_dict())
            best_step = step_index
        if current_metrics["self_acc"] <= 0.0:
            break
        selected_metrics = current_metrics

    model.load_state_dict(best_state_dict)
    reached_target = target_step is not None
    selected_metrics = best_metrics
    result = {
        **selected_metrics,
        "seed": trial_seed,
        "method": method_name,
        "method_label": METHOD_SPECS[method_name]["label"],
        "param_ratio": param_ratio,
        "status": "ok",
        "reason": None,
        "selected_params": selected_params,
        "reached_target": reached_target,
        "target_step": target_step,
        "best_step": best_step,
        "before_self_acc": before_metrics["self_acc"],
        "before_retain_acc": before_metrics["retain_acc"],
        "before_score": before_metrics["score"],
        "retain_acc_drop": before_metrics["retain_acc"]
        - selected_metrics["retain_acc"],
        "self_acc_drop": before_metrics["self_acc"] - selected_metrics["self_acc"],
        "gif_max_self_acc_for_selection": (
            args.gif_max_self_acc_for_selection if method_name == "gif" else None
        ),
    }
    status = "Target" if reached_target else "BestScore"
    print(
        format_metrics(
            f"{format_trial_prefix(trial_seed, method_name)} {status}:",
            result,
        )
    )
    return result


def summarize_results(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float | None], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((row["method"], row["param_ratio"]), []).append(row)

    summary: list[dict[str, object]] = []
    for (method_name, param_ratio), method_rows in grouped.items():
        successful = [row for row in method_rows if row["status"] == "ok"]
        skipped = [row for row in method_rows if row["status"] == "skipped"]
        summary_row: dict[str, object] = {
            "method": method_name,
            "method_label": METHOD_SPECS[method_name]["label"],
            "param_ratio": param_ratio,
            "num_trials": len(method_rows),
            "num_successful": len(successful),
            "num_skipped": len(skipped),
        }
        if successful:
            summary_row.update(
                {
                    "mean_retain_acc": float(
                        np.mean([row["retain_acc"] for row in successful])
                    ),
                    "mean_self_acc": float(
                        np.mean([row["self_acc"] for row in successful])
                    ),
                    "mean_score": float(np.mean([row["score"] for row in successful])),
                    "mean_retain_acc_drop": float(
                        np.mean([row["retain_acc_drop"] for row in successful])
                    ),
                    "reached_target_rate": float(
                        np.mean([float(row["reached_target"]) for row in successful])
                    ),
                }
            )
        else:
            summary_row.update(
                {
                    "mean_retain_acc": None,
                    "mean_self_acc": None,
                    "mean_score": None,
                    "mean_retain_acc_drop": None,
                    "reached_target_rate": None,
                }
            )
        if skipped:
            summary_row["skip_reasons"] = sorted(
                {str(row["reason"]) for row in skipped if row["reason"] is not None}
            )
        summary.append(summary_row)

    summary.sort(
        key=lambda item: (
            item["num_successful"],
            -1.0 if item["mean_score"] is None else item["mean_score"],
            -1.0 if item["mean_retain_acc"] is None else item["mean_retain_acc"],
        ),
        reverse=True,
    )
    return summary


def evaluate_retrained_baseline(
    args: argparse.Namespace,
    bundle,
    criterion: nn.Module,
    eval_target_ids: torch.Tensor,
    eval_target_masks: torch.Tensor,
    eval_target_targets: torch.Tensor,
    eval_retain_ids: torch.Tensor,
    eval_retain_masks: torch.Tensor,
    eval_retain_targets: torch.Tensor,
    device: torch.device,
) -> dict[str, object]:
    retrained_checkpoint = build_retrained_checkpoint_path(args)
    if not retrained_checkpoint.is_file():
        raise FileNotFoundError(
            "Retrained checkpoint not found: "
            f"{retrained_checkpoint}. Expected a target-label-removed retraining checkpoint."
        )

    model = build_model(bundle, args).to(device)
    load_checkpoint(model, retrained_checkpoint, device)
    model.eval()
    metrics = evaluate_unlearning_cached(
        model,
        eval_target_ids,
        eval_target_masks,
        eval_target_targets,
        eval_retain_ids,
        eval_retain_masks,
        eval_retain_targets,
        criterion,
        device,
        args.batch_size,
    )
    return {"checkpoint": str(retrained_checkpoint), "metrics": metrics}


def run_single_trial(
    args: argparse.Namespace,
    dataset_name: str,
    result_dir_name: str,
    trial_seed: int,
    save_dir: Path,
    retrained_baseline: dict[str, object],
) -> dict[str, object]:
    set_seed(trial_seed)
    device = torch.device(args.device)
    trajectory_dir = build_trajectory_path(args)

    bundle = create_hf_data_bundle(
        dataset_name,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        validation=True,
        seed=trial_seed,
        dataset_id=args.dataset_id,
        max_text_length=args.max_text_length,
        max_vocab_size=args.max_vocab_size,
        min_token_freq=args.min_token_freq,
    )
    criterion = nn.CrossEntropyLoss()
    base_model = build_model(bundle, args).to(device)
    load_checkpoint(base_model, args.checkpoint, device)
    base_model.eval()
    base_state_dict = deepcopy(base_model.state_dict())

    train_loader = bundle.train_loader
    test_loader = bundle.test_loader
    all_target_ids, all_target_masks, all_target_targets = collect_target_examples(
        test_loader, args.target_label
    )
    sampled_ids, sampled_masks, sampled_targets = sample_target_batches(
        all_target_ids,
        all_target_masks,
        all_target_targets,
        args.batch_size,
        args.num_target_batches,
    )
    retained_ids, retained_masks, retained_targets = collect_retained_examples(
        test_loader, args.target_label, 1
    )
    (
        eval_target_ids,
        eval_target_masks,
        eval_target_targets,
        eval_retain_ids,
        eval_retain_masks,
        eval_retain_targets,
    ) = collect_eval_splits(test_loader, args.target_label)

    model_factory = lambda: build_model(bundle, args)
    results: list[dict[str, object]] = []
    for method_name in args.methods:
        for param_ratio in method_param_ratios(method_name, args):
            results.append(
                run_single_method(
                    method_name=method_name,
                    param_ratio=param_ratio,
                    args=args,
                    trial_seed=trial_seed,
                    criterion=criterion,
                    train_loader=train_loader,
                    sampled_ids=sampled_ids,
                    sampled_masks=sampled_masks,
                    sampled_targets=sampled_targets,
                    all_target_count=len(all_target_targets),
                    retained_ids=retained_ids,
                    retained_masks=retained_masks,
                    retained_targets=retained_targets,
                    eval_target_ids=eval_target_ids,
                    eval_target_masks=eval_target_masks,
                    eval_target_targets=eval_target_targets,
                    eval_retain_ids=eval_retain_ids,
                    eval_retain_masks=eval_retain_masks,
                    eval_retain_targets=eval_retain_targets,
                    base_state_dict=base_state_dict,
                    model_factory=model_factory,
                    device=device,
                    trajectory_dir=trajectory_dir,
                )
            )

    ranked = sorted(
        results,
        key=lambda item: (
            item["status"] == "ok",
            item["reached_target"],
            item["retain_acc"],
            -item["target_step"] if item["target_step"] is not None else float("-inf"),
        ),
        reverse=True,
    )

    print("\nTop results")
    for row in ranked[: min(10, len(ranked))]:
        print(
            f"{row['method']:>18} | status={row['status']} | reached_target={int(row['reached_target'])} | "
            f"retain_acc={row['retain_acc']:.2f}% | self_acc={row['self_acc']:.2f}% | "
            f"target_step={row['target_step']} | selected={row['selected_params']} | seed={row['seed']}"
        )

    payload: dict[str, object] = {
        "config": {
            "seed": trial_seed,
            "dataset": dataset_name,
            "model": "text_transformer",
            "checkpoint": str(args.checkpoint),
            "retrained_checkpoint": str(retrained_baseline["checkpoint"]),
            "trajectory_dir": None if trajectory_dir is None else str(trajectory_dir),
            "tracin_max_checkpoints": args.tracin_max_checkpoints,
            "target_label": args.target_label,
            "methods": args.methods,
            "param_ratios": args.param_ratios,
            "tol": args.tol,
            "mu": args.mu,
            "max_iter": args.max_iter,
            "hypeinf_max_iter": args.hypeinf_max_iter,
            "edit_scale": args.edit_scale,
            "max_update_steps": args.max_update_steps,
            "gif_max_self_acc_for_selection": args.gif_max_self_acc_for_selection,
            "num_target_batches": args.num_target_batches,
            "device": args.device,
        },
        "retrained_baseline": retrained_baseline,
        "results": ranked,
    }
    trial_path = save_dir / f"seed_{trial_seed:04d}.json"
    trial_path.parent.mkdir(parents=True, exist_ok=True)
    trial_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved results to {trial_path}")
    return payload


def main_for_dataset(
    *,
    dataset_name: str,
    result_dir_name: str,
    default_checkpoint: Path,
    default_target_label: int,
) -> None:
    args = parse_args(default_checkpoint, default_target_label)
    if args.num_trials <= 0:
        raise ValueError("--num-trials must be positive.")

    set_seed(args.seed)
    device = torch.device(args.device)

    base_bundle = create_hf_data_bundle(
        dataset_name,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        validation=True,
        seed=args.seed,
        dataset_id=args.dataset_id,
        max_text_length=args.max_text_length,
        max_vocab_size=args.max_vocab_size,
        min_token_freq=args.min_token_freq,
    )
    base_model = build_model(base_bundle, args).to(device)
    load_checkpoint(base_model, args.checkpoint, device)
    base_model.eval()
    base_criterion = nn.CrossEntropyLoss()
    (
        base_eval_target_ids,
        base_eval_target_masks,
        base_eval_target_targets,
        base_eval_retain_ids,
        base_eval_retain_masks,
        base_eval_retain_targets,
    ) = collect_eval_splits(base_bundle.test_loader, args.target_label)
    base_before_metrics = evaluate_unlearning_cached(
        base_model,
        base_eval_target_ids,
        base_eval_target_masks,
        base_eval_target_targets,
        base_eval_retain_ids,
        base_eval_retain_masks,
        base_eval_retain_targets,
        base_criterion,
        device,
        args.batch_size,
    )
    print(format_metrics("[global] Before:", base_before_metrics))
    retrained_baseline = evaluate_retrained_baseline(
        args=args,
        bundle=base_bundle,
        criterion=base_criterion,
        eval_target_ids=base_eval_target_ids,
        eval_target_masks=base_eval_target_masks,
        eval_target_targets=base_eval_target_targets,
        eval_retain_ids=base_eval_retain_ids,
        eval_retain_masks=base_eval_retain_masks,
        eval_retain_targets=base_eval_retain_targets,
        device=device,
    )
    print(format_metrics("[global] Retrained:", retrained_baseline["metrics"]))

    trajectory_dir = build_trajectory_path(args)
    if trajectory_dir is not None:
        print(f"[global] Trajectory dir: {trajectory_dir}")

    save_root = EXPERIMENT_ROOT / "results" / result_dir_name
    all_rows: list[dict[str, object]] = []
    for trial_index in range(args.num_trials):
        trial_seed = args.seed + trial_index
        payload = run_single_trial(
            args=args,
            dataset_name=dataset_name,
            result_dir_name=result_dir_name,
            trial_seed=trial_seed,
            save_dir=save_root,
            retrained_baseline=retrained_baseline,
        )
        all_rows.extend(payload["results"])

    summary = summarize_results(all_rows)
    print("\nSummary")
    for row in summary[: min(10, len(summary))]:
        print(
            f"{row['method']:>18} | success={row['num_successful']}/{row['num_trials']} | "
            f"mean_retain_acc={row['mean_retain_acc']} | "
            f"mean_self_acc={row['mean_self_acc']} | mean_score={row['mean_score']}"
        )
