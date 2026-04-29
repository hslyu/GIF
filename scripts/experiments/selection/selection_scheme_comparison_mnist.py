#!/usr/bin/env python3
"""Compare parameter selection schemes under a fixed generalized-influence update."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

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

from _mnist_unlearning_common import (  # noqa: E402
    build_total_loss,
    collect_retained_examples,
    collect_target_examples,
    evaluate_unlearning,
    format_metrics,
    load_checkpoint,
    set_seed,
)

from gif.data.mnist import MNISTDataLoader  # noqa: E402
from gif.influence import generalized_influence  # noqa: E402
from gif.models import FullyConnectedNet, ResNet18, ResNet34  # noqa: E402
from gif.selection import (  # noqa: E402
    CAPS,
    EKFACCAPS,
    HighestKGradients,
    HighestKOutputs,
    LowestKGradients,
    LowestKOutputs,
    Random,
    ReverseCAPS,
)

SELECTION_REGISTRY = {
    "caps": CAPS,
    "ekfac_caps": EKFACCAPS,
    "reverse_caps": ReverseCAPS,
    "highest_k_outputs": HighestKOutputs,
    "highest_k_gradients": HighestKGradients,
    "lowest_k_outputs": LowestKOutputs,
    "lowest_k_gradients": LowestKGradients,
    "random": Random,
}

GRADIENT_SELECTOR_NAMES = {
    "highest_k_gradients",
    "lowest_k_gradients",
}

CAPS_STYLE_SELECTOR_NAMES = {"caps", "ekfac_caps", "reverse_caps"}
EMPTY_SELECTION_ERROR_MARKERS = (
    "No blocks were selected",
    "empty parameter subset",
)
DEFAULT_PARAM_RATIOS = [
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    1.00,
]
DEFAULT_SELECTORS = [
    # "caps",
    # "reverse_caps",
    # "highest_k_outputs",
    # "highest_k_gradients",
    "lowest_k_outputs",
    "lowest_k_gradients",
    # "random",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare selection schemes on a fixed generalized-influence unlearning update."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument(
        "--model",
        choices=["fcn", "resnet18", "resnet34"],
        default="fcn",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--num-target-batches", type=int, default=2)
    parser.add_argument(
        "--selectors",
        nargs="+",
        choices=sorted(SELECTION_REGISTRY.keys()),
        default=DEFAULT_SELECTORS,
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
    parser.add_argument("--edit-scale", type=float, default=0.1)
    parser.add_argument("--max-update-steps", type=int, default=200)
    parser.add_argument("--target-self-acc", type=float, default=0.1)
    parser.add_argument("--caps-lam", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--dropout-prob", type=float, default=0.1)
    parser.add_argument("--save-json", type=Path, default=None)
    parser.add_argument(
        "--seed-output-dir",
        type=Path,
        default=None,
        help="Directory for per-seed seed_XXXX.json files.",
    )
    return parser.parse_args()


def build_model(args: argparse.Namespace) -> tuple[nn.Module, bool]:
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
    if args.model == "resnet18":
        return ResNet18(in_channels=1), False
    if args.model == "resnet34":
        return ResNet34(in_channels=1), False
    raise ValueError(f"Unsupported model: {args.model}")


def build_checkpoint_path(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        return args.checkpoint
    if args.model == "fcn":
        return PROJECT_ROOT / "checkpoints" / "mnist_fcn_deep.pth"
    return PROJECT_ROOT / "checkpoints" / f"mnist_{args.model}.pth"


def default_save_path(args: argparse.Namespace) -> Path:
    return (
        EXPERIMENT_ROOT
        / "results"
        / args.model
        / f"seed_{args.seed:04d}_trials_{args.num_trials:03d}.json"
    )


def build_selector(
    selector_name: str,
    model: nn.Module,
    ratio: float,
    selector_inputs: torch.Tensor,
    selector_targets: torch.Tensor,
    retained_inputs: torch.Tensor,
    retained_targets: torch.Tensor,
    criterion: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
):
    selector_cls = SELECTION_REGISTRY[selector_name]
    if selector_name in CAPS_STYLE_SELECTOR_NAMES:
        selector = selector_cls(
            model,
            ratio=ratio,
            lam=args.caps_lam,
        )
        target_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(selector_inputs, selector_targets),
            batch_size=min(len(selector_inputs), args.batch_size),
            shuffle=False,
        )
        retained_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(retained_inputs, retained_targets),
            batch_size=min(len(retained_inputs), args.batch_size),
            shuffle=False,
        )
        selector.fit(
            target_loader=target_loader,
            retained_loader=retained_loader,
            criterion=criterion,
            device=device,
        )
        return selector

    selector = selector_cls(model, ratio)
    selector.register_hooks()
    model.zero_grad(set_to_none=True)
    selector_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(selector_inputs, selector_targets),
        batch_size=min(len(selector_inputs), args.batch_size),
        shuffle=False,
    )
    total_selector_examples = len(selector_inputs)
    for batch_inputs, batch_targets in selector_loader:
        moved_inputs = batch_inputs.to(device)
        moved_targets = batch_targets.to(device)
        if selector_name in GRADIENT_SELECTOR_NAMES:
            target_loss = criterion(model(moved_inputs), moved_targets)
            target_loss = target_loss * (
                len(batch_targets) / total_selector_examples
            )
            target_loss.backward()
            model.zero_grad(set_to_none=True)
        else:
            with torch.no_grad():
                model(moved_inputs)
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


def is_empty_selection_error(error: RuntimeError) -> bool:
    message = str(error)
    return any(marker in message for marker in EMPTY_SELECTION_ERROR_MARKERS)


def run_single_selector(
    selector_name: str,
    param_ratio: float,
    args: argparse.Namespace,
    trial_index: int,
    trial_seed: int,
    criterion: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    selector_inputs: torch.Tensor,
    selector_targets: torch.Tensor,
    sampled_inputs: torch.Tensor,
    sampled_targets: torch.Tensor,
    all_target_count: int,
    retained_inputs: torch.Tensor,
    retained_targets: torch.Tensor,
    base_state_dict: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, float]:
    model, _ = build_model(args)
    model = model.to(device)
    model.load_state_dict(base_state_dict)
    model.eval()

    before_metrics = evaluate_unlearning(
        model, test_loader, criterion, args.target_label, device
    )

    total_loss = build_total_loss(
        model,
        retained_inputs,
        retained_targets,
        criterion,
        device,
    )
    target_scaling = all_target_count / (len(train_loader.dataset) - all_target_count)
    selector = build_selector(
        selector_name=selector_name,
        model=model,
        ratio=param_ratio,
        selector_inputs=selector_inputs,
        selector_targets=selector_targets,
        retained_inputs=retained_inputs,
        retained_targets=retained_targets,
        criterion=criterion,
        device=device,
        args=args,
    )
    index_list = selector.get_parameters()
    if len(index_list) == 0:
        raise RuntimeError(
            f"{selector_name} returned an empty parameter subset for ratio={param_ratio}."
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
        mu=args.mu,
        tol=args.tol,
        max_iter=args.max_iter,
        verbose=False,
    )
    norm = torch.norm(influence)
    if torch.isnan(norm) or norm.item() == 0.0:
        raise RuntimeError(
            f"{selector_name} produced a zero or NaN influence vector for ratio={param_ratio}."
        )
    normalized_update = influence / norm

    target_metrics = None
    target_state = None
    target_step = None
    for step in range(1, args.max_update_steps + 1):
        selector.update_network(normalized_update * args.edit_scale)
        current_metrics = evaluate_unlearning(
            model, test_loader, criterion, args.target_label, device
        )
        if current_metrics["self_acc"] <= args.target_self_acc:
            target_metrics = current_metrics
            target_state = deepcopy(model.state_dict())
            target_step = step
            break

    reached_target = target_metrics is not None
    if reached_target:
        model.load_state_dict(target_state)
        selected_metrics = target_metrics
    else:
        selected_metrics = evaluate_unlearning(
            model, test_loader, criterion, args.target_label, device
        )

    retain_acc = selected_metrics["retain_acc"]
    self_acc = selected_metrics["self_acc"]
    selected_metrics = {
        **selected_metrics,
        "seed": trial_seed,
        "trial": trial_index,
        "selector": selector_name,
        "param_ratio": param_ratio,
        "selected_params": int(len(index_list)),
        "reached_target": reached_target,
        "target_step": target_step,
        "target_self_acc_threshold": args.target_self_acc,
        "before_self_acc": before_metrics["self_acc"],
        "before_retain_acc": before_metrics["retain_acc"],
        "before_score": before_metrics["score"],
        "retain_acc_drop": before_metrics["retain_acc"] - retain_acc,
        "self_acc_drop": before_metrics["self_acc"] - self_acc,
    }
    status = "Target" if reached_target else "Missed"
    print(
        format_metrics(
            f"[trial={trial_index} seed={trial_seed} {selector_name} ratio={param_ratio:.3f}] {status}:",
            selected_metrics,
        )
    )
    return selected_metrics


def run_single_trial(
    args: argparse.Namespace,
    trial_index: int,
    trial_seed: int,
    save_dir: Path,
) -> dict[str, object]:
    set_seed(trial_seed)
    device = torch.device(args.device)

    model, flatten = build_model(args)
    model = model.to(device)
    load_checkpoint(model, build_checkpoint_path(args), device)
    model.eval()
    base_state_dict = deepcopy(model.state_dict())

    criterion = nn.CrossEntropyLoss()
    data_loader = MNISTDataLoader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        validation=False,
        flatten=flatten,
        root=str(args.data_root),
    )
    train_loader, test_loader = data_loader.get_data_loaders()
    all_target_inputs, all_target_targets = collect_target_examples(
        test_loader, args.target_label
    )
    sampled_inputs, sampled_targets = sample_target_batches(
        all_target_inputs,
        all_target_targets,
        args.batch_size,
        args.num_target_batches,
    )
    retained_inputs, retained_targets = collect_retained_examples(
        test_loader,
        args.target_label,
        1,
    )

    results: list[dict[str, float]] = []
    for selector_name in args.selectors:
        for param_ratio in args.param_ratios:
            try:
                metrics = run_single_selector(
                    selector_name=selector_name,
                    param_ratio=param_ratio,
                    args=args,
                    trial_index=trial_index,
                    trial_seed=trial_seed,
                    criterion=criterion,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    selector_inputs=all_target_inputs,
                    selector_targets=all_target_targets,
                    sampled_inputs=sampled_inputs,
                    sampled_targets=sampled_targets,
                    all_target_count=len(all_target_inputs),
                    retained_inputs=retained_inputs,
                    retained_targets=retained_targets,
                    base_state_dict=base_state_dict,
                    device=device,
                )
            except RuntimeError as error:
                if is_empty_selection_error(error):
                    print(
                        f"[trial={trial_index} seed={trial_seed} {selector_name} "
                        f"ratio={param_ratio:.3f}] Skipped empty selection: {error}"
                    )
                    continue
                raise
            results.append(metrics)

    ranked = sorted(
        results,
        key=lambda item: (
            item["reached_target"],
            item["retain_acc"],
            -item["target_step"] if item["target_step"] is not None else float("-inf"),
        ),
        reverse=True,
    )

    print("\nTop results")
    for row in ranked[: min(10, len(ranked))]:
        print(
            f"{row['selector']:>22} | ratio={row['param_ratio']:.3f} | "
            f"reached_target={int(row['reached_target'])} | "
            f"retain_acc={row['retain_acc']:.2f}% | "
            f"self_acc={row['self_acc']:.2f}% | "
            f"target_step={row['target_step']} | "
            f"selected={row['selected_params']} | "
            f"trial={row['trial']} | seed={row['seed']}"
        )

    payload: dict[str, object] = {
        "config": {
            "seed": trial_seed,
            "model": args.model,
            "checkpoint": str(build_checkpoint_path(args)),
            "target_label": args.target_label,
            "selectors": args.selectors,
            "param_ratios": args.param_ratios,
            "tol": args.tol,
            "mu": args.mu,
            "max_iter": args.max_iter,
            "edit_scale": args.edit_scale,
            "max_update_steps": args.max_update_steps,
            "num_target_batches": args.num_target_batches,
            "device": args.device,
        },
        "results": ranked,
    }
    trial_path = save_dir / f"seed_{trial_seed:04d}.json"
    trial_path.parent.mkdir(parents=True, exist_ok=True)
    trial_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved results to {trial_path}")
    return payload


def main() -> None:
    args = parse_args()
    if args.num_trials <= 0:
        raise ValueError("--num-trials must be positive.")

    set_seed(args.seed)
    device = torch.device(args.device)
    base_model, flatten = build_model(args)
    base_model = base_model.to(device)
    load_checkpoint(base_model, build_checkpoint_path(args), device)
    base_model.eval()
    base_criterion = nn.CrossEntropyLoss()
    base_data_loader = MNISTDataLoader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        validation=False,
        flatten=flatten,
        root=str(args.data_root),
    )
    _, base_test_loader = base_data_loader.get_data_loaders()
    base_before_metrics = evaluate_unlearning(
        base_model, base_test_loader, base_criterion, args.target_label, device
    )
    print(format_metrics("[global] Before:", base_before_metrics))

    save_root = args.seed_output_dir or EXPERIMENT_ROOT / "results" / args.model
    aggregate_runs: list[dict[str, object]] = []
    for trial_index in range(args.num_trials):
        trial_seed = args.seed + trial_index
        aggregate_runs.append(
            run_single_trial(
                args=args,
                trial_index=trial_index,
                trial_seed=trial_seed,
                save_dir=save_root,
            )
        )

    del aggregate_runs


if __name__ == "__main__":
    main()
