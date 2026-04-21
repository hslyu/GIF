#!/usr/bin/env python3
"""
Minimal MNIST unlearning script using GIF only.

This script is a plain-Python port of the GIF path from
`GIF_reference/scripts/table2-3-IF_comparison_mnist_0.ipynb`.
It expects a pretrained MNIST `ResNet18(in_channels=1)` checkpoint and applies
Generalized Influence Functions to forget one target label while retaining
performance on the remaining labels.
"""

from __future__ import annotations

import argparse
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from gif.data.mnist import MNISTDataLoader
from gif.influence import generalized_influence
from gif.models import ResNet18
from gif.selection import CAPS, HighestKGradients


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a minimal GIF-based MNIST unlearning experiment."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--save-path", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-target-samples", type=int, default=256)
    parser.add_argument("--num-retain-batches", type=int, default=1)
    parser.add_argument("--param-ratio", type=float, default=0.05)
    parser.add_argument(
        "--schemes",
        nargs="+",
        choices=["caps", "highest_k_gradients", "tracin", "hyperinf", "datainf"],
        default=["caps", "highest_k_gradients"],
    )
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        default=None,
        help="Directory containing epoch-wise checkpoints for TracIn.",
    )
    parser.add_argument("--caps-lam", type=float, default=1e-6)
    parser.add_argument("--caps-min-curv", type=float, default=1e-12)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--mu", type=float, default=3.0)
    parser.add_argument("--hyperinf-beta-scale", type=float, default=0.9)
    parser.add_argument("--datainf-damping", type=float, default=1e-6)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--edit-scale", type=float, default=0.03)
    parser.add_argument("--max-update-steps", type=int, default=25)
    parser.add_argument("--min-retain-acc", type=float, default=98.7)
    parser.add_argument(
        "--target-self-acc",
        type=float,
        default=0.2,
        help="Stop when target self accuracy drops below this percentage value. Example: 0.2 means 0.2%%.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(
    model: torch.nn.Module, checkpoint_path: Path, device: torch.device
) -> None:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["net"])


def save_checkpoint(model: torch.nn.Module, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"net": model.state_dict()}, output_path)


def evaluate_split(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    label: int,
    include_label: bool,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            mask = targets == label if include_label else targets != label
            if not torch.any(mask):
                continue

            inputs = inputs[mask].to(device)
            targets = targets[mask].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * targets.size(0)
            total_correct += outputs.argmax(dim=1).eq(targets).sum().item()
            total_examples += targets.size(0)

    if total_examples == 0:
        raise RuntimeError("Split evaluation received zero examples.")

    mean_loss = total_loss / total_examples
    accuracy = 100.0 * total_correct / total_examples
    return mean_loss, accuracy


def evaluate_unlearning(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    target_label: int,
    device: torch.device,
) -> dict[str, float]:
    self_loss, self_acc = evaluate_split(
        model, dataloader, criterion, target_label, True, device
    )
    retain_loss, retain_acc = evaluate_split(
        model, dataloader, criterion, target_label, False, device
    )
    score = f1_unlearning_score(self_acc, retain_acc)
    return {
        "self_loss": self_loss,
        "self_acc": self_acc,
        "retain_loss": retain_loss,
        "retain_acc": retain_acc,
        "score": score,
    }


def f1_unlearning_score(self_acc: float, retain_acc: float) -> float:
    self_acc /= 100.0
    retain_acc /= 100.0
    if self_acc == 1.0 and retain_acc == 0.0:
        return 0.0
    return 2.0 * (1.0 - self_acc) * retain_acc / (1.0 - self_acc + retain_acc)


def build_total_loss(
    model: torch.nn.Module,
    retained_inputs: torch.Tensor,
    retained_targets: torch.Tensor,
    criterion: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    if len(retained_inputs) == 0:
        raise RuntimeError(
            "Could not build retained-data loss from an empty retained set."
        )
    return criterion(model(retained_inputs.to(device)), retained_targets.to(device))


def collect_target_examples(
    dataloader: torch.utils.data.DataLoader, target_label: int
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs_list = []
    targets_list = []
    for inputs, targets in dataloader:
        mask = targets == target_label
        if torch.any(mask):
            inputs_list.append(inputs[mask])
            targets_list.append(targets[mask])

    if not inputs_list:
        raise RuntimeError(f"No examples found for target label {target_label}.")

    return torch.cat(inputs_list, dim=0), torch.cat(targets_list, dim=0)


def collect_retained_examples(
    dataloader: torch.utils.data.DataLoader,
    target_label: int,
    num_batches: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs_list = []
    targets_list = []
    used_batches = 0
    for inputs, targets in dataloader:
        if used_batches >= num_batches:
            break
        mask = targets != target_label
        if not torch.any(mask):
            continue
        inputs_list.append(inputs[mask])
        targets_list.append(targets[mask])
        used_batches += 1

    if not inputs_list:
        raise RuntimeError("No retained examples found for the requested batches.")

    return torch.cat(inputs_list, dim=0), torch.cat(targets_list, dim=0)


def sample_target_batch(
    inputs: torch.Tensor, targets: torch.Tensor, num_samples: int
) -> tuple[torch.Tensor, torch.Tensor]:
    num_samples = min(num_samples, len(inputs))
    sampled_indices = np.random.choice(len(inputs), size=num_samples, replace=False)
    return inputs[sampled_indices], targets[sampled_indices]


def build_loader(
    inputs: torch.Tensor, targets: torch.Tensor, batch_size: int
) -> torch.utils.data.DataLoader:
    return DataLoader(
        TensorDataset(inputs, targets),
        batch_size=min(batch_size, len(inputs)),
        shuffle=False,
    )


def select_parameters(
    scheme: str,
    model: torch.nn.Module,
    criterion: nn.Module,
    sampled_inputs: torch.Tensor,
    sampled_targets: torch.Tensor,
    retained_inputs: torch.Tensor,
    retained_targets: torch.Tensor,
    target_scaling: float,
    param_ratio: float,
    caps_lam: float,
    caps_min_curv: float,
    batch_size: int,
    device: torch.device,
):
    if scheme == "caps":
        selector = CAPS(
            model,
            ratio=param_ratio,
            lam=caps_lam,
            min_curv=caps_min_curv,
        )
        target_loader = build_loader(sampled_inputs, sampled_targets, batch_size)
        retained_loader = build_loader(retained_inputs, retained_targets, batch_size)
        selector.fit(
            target_loader=target_loader,
            retained_loader=retained_loader,
            criterion=criterion,
            device=device,
        )
        return selector

    selector = HighestKGradients(model, param_ratio)
    selector.register_hooks()

    model.zero_grad(set_to_none=True)
    scaled_target_loss = (
        criterion(model(sampled_inputs.to(device)), sampled_targets.to(device))
        * target_scaling
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Full backward hook is firing when gradients are computed with respect to module outputs",
        )
        scaled_target_loss.backward()
    selector.remove_hooks()
    return selector


def compute_gif_update(
    scheme: str,
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    sampled_inputs: torch.Tensor,
    sampled_targets: torch.Tensor,
    all_target_count: int,
    retained_inputs: torch.Tensor,
    retained_targets: torch.Tensor,
    param_ratio: float,
    caps_lam: float,
    caps_min_curv: float,
    batch_size: int,
    tol: float,
    mu: float,
    max_iter: int,
    device: torch.device,
) -> tuple[object, torch.Tensor]:
    model.eval()
    total_loss = build_total_loss(
        model, retained_inputs, retained_targets, criterion, device
    )
    if len(train_loader.dataset) <= all_target_count:
        raise RuntimeError("Target set size must be smaller than the training dataset.")
    target_scaling = all_target_count / (len(train_loader.dataset) - all_target_count)
    selector = select_parameters(
        scheme=scheme,
        model=model,
        criterion=criterion,
        sampled_inputs=sampled_inputs,
        sampled_targets=sampled_targets,
        retained_inputs=retained_inputs,
        retained_targets=retained_targets,
        target_scaling=target_scaling,
        param_ratio=param_ratio,
        caps_lam=caps_lam,
        caps_min_curv=caps_min_curv,
        batch_size=batch_size,
        device=device,
    )
    index_list = selector.get_parameters()

    if len(index_list) == 0:
        raise RuntimeError(
            f"{scheme} parameter selection returned an empty index list."
        )

    model.zero_grad(set_to_none=True)
    target_loss = (
        criterion(model(sampled_inputs.to(device)), sampled_targets.to(device))
        * target_scaling
    )

    influence = generalized_influence(
        model,
        total_loss,
        target_loss,
        index_list,
        mu=mu,
        tol=tol,
        max_iter=max_iter,
        verbose=False,
    )
    norm = torch.norm(influence)
    if torch.isnan(norm) or norm.item() == 0.0:
        raise RuntimeError("GIF update has zero or NaN norm.")

    return selector, influence / norm


def format_metrics(prefix: str, metrics: dict[str, float]) -> str:
    return (
        f"{prefix} retain_acc={metrics['retain_acc']:.2f}% "
        f"retain_loss={metrics['retain_loss']:.4f} "
        f"self_acc={metrics['self_acc']:.2f}% "
        f"self_loss={metrics['self_loss']:.4f} "
        f"score={metrics['score']:.4f}"
    )


def format_summary_row(scheme: str, metrics: dict[str, float]) -> str:
    return (
        f"{scheme:>20} | "
        f"retain_acc={metrics['retain_acc']:.2f}% | "
        f"self_acc={metrics['self_acc']:.2f}% | "
        f"score={metrics['score']:.4f}"
    )


def run_single_scheme(
    scheme: str,
    args: argparse.Namespace,
    base_state_dict: dict[str, torch.Tensor],
    criterion: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    sampled_inputs: torch.Tensor,
    sampled_targets: torch.Tensor,
    all_target_count: int,
    retained_inputs: torch.Tensor,
    retained_targets: torch.Tensor,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, float]]:
    model = ResNet18(in_channels=1).to(device)
    model.load_state_dict(base_state_dict)
    model.eval()

    before_metrics = evaluate_unlearning(
        model, test_loader, criterion, args.target_label, device
    )
    print(format_metrics(f"[{scheme}] Before:", before_metrics))

    selector, normalized_update = compute_gif_update(
        scheme=scheme,
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        sampled_inputs=sampled_inputs,
        sampled_targets=sampled_targets,
        all_target_count=all_target_count,
        retained_inputs=retained_inputs,
        retained_targets=retained_targets,
        param_ratio=args.param_ratio,
        caps_lam=args.caps_lam,
        caps_min_curv=args.caps_min_curv,
        batch_size=args.batch_size,
        tol=args.tol,
        mu=args.mu,
        max_iter=args.max_iter,
        device=device,
    )

    best_metrics = before_metrics
    best_state = deepcopy(model.state_dict())

    for update_step in range(1, args.max_update_steps + 1):
        selector.update_network(normalized_update * args.edit_scale)
        current_metrics = evaluate_unlearning(
            model, test_loader, criterion, args.target_label, device
        )
        print(format_metrics(f"[{scheme}] Step {update_step}:", current_metrics))

        if current_metrics["score"] > best_metrics["score"]:
            best_metrics = current_metrics
            best_state = deepcopy(model.state_dict())

        if (
            current_metrics["retain_acc"] < args.min_retain_acc
            or current_metrics["self_acc"] < args.target_self_acc
        ):
            break

    model.load_state_dict(best_state)
    best_metrics = {
        **best_metrics,
        "before_self_loss": before_metrics["self_loss"],
        "before_self_acc": before_metrics["self_acc"],
        "before_retain_loss": before_metrics["retain_loss"],
        "before_retain_acc": before_metrics["retain_acc"],
        "before_score": before_metrics["score"],
        "retain_acc_drop": before_metrics["retain_acc"] - best_metrics["retain_acc"],
        "self_acc_drop": before_metrics["self_acc"] - best_metrics["self_acc"],
    }
    print(format_metrics(f"[{scheme}] Best:", best_metrics))
    return model, best_metrics


def run_experiment(args: argparse.Namespace) -> dict[str, dict[str, float]]:
    device = torch.device(args.device)
    set_seed(args.seed)

    model = ResNet18(in_channels=1).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    data_loader = MNISTDataLoader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        validation=False,
        root=str(args.data_root),
    )
    train_loader, test_loader = data_loader.get_data_loaders()
    all_target_inputs, all_target_targets = collect_target_examples(
        test_loader, args.target_label
    )
    sampled_inputs, sampled_targets = sample_target_batch(
        all_target_inputs, all_target_targets, args.num_target_samples
    )
    retained_inputs, retained_targets = collect_retained_examples(
        test_loader, args.target_label, args.num_retain_batches
    )

    base_state_dict = deepcopy(model.state_dict())
    results = {}
    best_models = {}
    for scheme in args.schemes:
        best_model, best_metrics = run_single_scheme(
            scheme=scheme,
            args=args,
            base_state_dict=base_state_dict,
            criterion=criterion,
            train_loader=train_loader,
            test_loader=test_loader,
            sampled_inputs=sampled_inputs,
            sampled_targets=sampled_targets,
            all_target_count=len(all_target_inputs),
            retained_inputs=retained_inputs,
            retained_targets=retained_targets,
            device=device,
        )
        best_models[scheme] = best_model
        results[scheme] = best_metrics

    print("\nComparison")
    for scheme in args.schemes:
        print(format_summary_row(scheme, results[scheme]))

    if args.save_path is not None:
        for scheme in args.schemes:
            if len(args.schemes) == 1:
                output_path = args.save_path
            else:
                output_path = args.save_path.with_name(
                    f"{args.save_path.stem}_{scheme}{args.save_path.suffix}"
                )
            save_checkpoint(best_models[scheme], output_path)
            print(f"Saved {scheme} checkpoint to {output_path}")

    return results


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
