#!/usr/bin/env python3
"""Compare parameter selection schemes on CIFAR-10 ResNet18 unlearning."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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
    format_metrics,
    load_checkpoint,
    set_seed,
)

from gif.data.huggingface import create_hf_data_bundle  # noqa: E402
from gif.influence import generalized_influence  # noqa: E402
from gif.models import ResNet18  # noqa: E402
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
    "highest_k_outputs",
    "highest_k_gradients",
    "lowest_k_outputs",
    "lowest_k_gradients",
    "caps",
    # "ekfac_caps",
    "reverse_caps",
    "random",
]
DEFAULT_SELECTORS = [
    selector for selector in DEFAULT_SELECTORS if selector in SELECTION_REGISTRY
]
if not DEFAULT_SELECTORS:
    DEFAULT_SELECTORS = sorted(SELECTION_REGISTRY.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare selection schemes on CIFAR-10 ResNet18 with a fixed generalized-influence update."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "hf_cifar10_resnet18.pth",
    )
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--num-target-batches", type=int, default=10)
    parser.add_argument("--dataset-id", type=str, default=None)
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
    parser.add_argument("--edit-scale", type=float, default=0.02)
    parser.add_argument("--max-update-steps", type=int, default=200)
    parser.add_argument("--target-self-acc", type=float, default=0.1)
    parser.add_argument("--caps-lam", type=float, default=1e-4)
    parser.add_argument(
        "--cache-test-on-device",
        action="store_true",
        default=False,
        help="Cache CIFAR-10 evaluation tensors on the target device.",
    )
    parser.add_argument(
        "--no-cache-test-on-device",
        dest="cache_test_on_device",
        action="store_false",
        help="Keep evaluation tensors on CPU.",
    )
    parser.add_argument("--save-json", type=Path, default=None)
    parser.add_argument(
        "--seed-output-dir",
        type=Path,
        default=None,
        help="Directory for per-seed seed_XXXX.json files.",
    )
    return parser.parse_args()


def default_save_path(args: argparse.Namespace) -> Path:
    return (
        EXPERIMENT_ROOT
        / "results"
        / "cifar10_resnet18"
        / f"seed_{args.seed:04d}_trials_{args.num_trials:03d}.json"
    )


def result_key(row: dict[str, object]) -> tuple[str, float]:
    return str(row["selector"]), float(row["param_ratio"])


def rank_results(results: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        results,
        key=lambda item: (
            item["reached_target"],
            item["retain_acc"],
            -item["target_step"] if item["target_step"] is not None else float("-inf"),
        ),
        reverse=True,
    )


def merge_trial_payload(
    trial_path: Path,
    trial_payload: dict[str, object],
) -> tuple[dict[str, object], bool]:
    if not trial_path.exists():
        return trial_payload, False

    existing_payload = json.loads(trial_path.read_text())
    existing_results = {
        result_key(row): row for row in existing_payload.get("results", [])
    }

    for row in trial_payload["results"]:
        existing_results[result_key(row)] = row

    merged_payload = {
        **existing_payload,
        "config": {
            **existing_payload.get("config", {}),
            **trial_payload["config"],
        },
        "results": rank_results(list(existing_results.values())),
    }
    return merged_payload, True


def build_model() -> nn.Module:
    return ResNet18(in_channels=3)


def _move_tensor(
    tensor: torch.Tensor,
    device: torch.device,
    *,
    non_blocking: bool = True,
) -> torch.Tensor:
    if tensor.device == device:
        return tensor
    return tensor.to(device, non_blocking=non_blocking)


def is_empty_selection_error(error: RuntimeError) -> bool:
    message = str(error)
    return any(marker in message for marker in EMPTY_SELECTION_ERROR_MARKERS)


def build_cached_loader(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
) -> DataLoader:
    return DataLoader(
        TensorDataset(inputs, targets),
        batch_size=min(batch_size, len(inputs)),
        shuffle=False,
        pin_memory=inputs.device.type == "cpu",
    )


def collect_eval_splits(
    dataloader: DataLoader,
    target_label: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    target_inputs = []
    target_targets = []
    retain_inputs = []
    retain_targets = []
    for inputs, targets in dataloader:
        target_mask = targets == target_label
        retain_mask = ~target_mask
        if torch.any(target_mask):
            target_inputs.append(inputs[target_mask])
            target_targets.append(targets[target_mask])
        if torch.any(retain_mask):
            retain_inputs.append(inputs[retain_mask])
            retain_targets.append(targets[retain_mask])

    if not target_inputs or not retain_inputs:
        raise RuntimeError("Could not build CIFAR-10 evaluation splits.")

    return (
        torch.cat(target_inputs, dim=0),
        torch.cat(target_targets, dim=0),
        torch.cat(retain_inputs, dim=0),
        torch.cat(retain_targets, dim=0),
    )


def sample_target_batches(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
    num_batches: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_samples = min(batch_size * num_batches, len(inputs))
    permutation = torch.randperm(len(targets))[:num_samples]
    return inputs[permutation], targets[permutation]


def evaluate_split_tensors(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    device: torch.device,
    batch_size: int,
) -> tuple[float, float]:
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            batch_inputs = _move_tensor(inputs[start : start + batch_size], device)
            batch_targets = _move_tensor(targets[start : start + batch_size], device)
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            total_loss += loss.item() * batch_targets.size(0)
            total_correct += outputs.argmax(dim=1).eq(batch_targets).sum().item()
            total_examples += batch_targets.size(0)

    if total_examples == 0:
        raise RuntimeError("Split evaluation received zero examples.")

    mean_loss = total_loss / total_examples
    accuracy = 100.0 * total_correct / total_examples
    return mean_loss, accuracy


def f1_unlearning_score(self_acc: float, retain_acc: float) -> float:
    self_acc /= 100.0
    retain_acc /= 100.0
    if self_acc == 1.0 and retain_acc == 0.0:
        return 0.0
    return 2.0 * (1.0 - self_acc) * retain_acc / (1.0 - self_acc + retain_acc)


def evaluate_unlearning_cached(
    model: nn.Module,
    target_inputs: torch.Tensor,
    target_targets: torch.Tensor,
    retain_inputs: torch.Tensor,
    retain_targets: torch.Tensor,
    criterion: nn.Module,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    self_loss, self_acc = evaluate_split_tensors(
        model, target_inputs, target_targets, criterion, device, batch_size
    )
    retain_loss, retain_acc = evaluate_split_tensors(
        model, retain_inputs, retain_targets, criterion, device, batch_size
    )
    return {
        "self_loss": self_loss,
        "self_acc": self_acc,
        "retain_loss": retain_loss,
        "retain_acc": retain_acc,
        "score": f1_unlearning_score(self_acc, retain_acc),
    }


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
        target_loader = build_cached_loader(
            selector_inputs, selector_targets, args.batch_size
        )
        retained_loader = build_cached_loader(
            retained_inputs, retained_targets, args.batch_size
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
    selector_loader = build_cached_loader(
        selector_inputs, selector_targets, args.batch_size
    )
    total_selector_examples = len(selector_targets)
    for batch_inputs, batch_targets in selector_loader:
        moved_inputs = _move_tensor(batch_inputs, device)
        moved_targets = _move_tensor(batch_targets, device)
        if selector_name in GRADIENT_SELECTOR_NAMES:
            target_loss = criterion(model(moved_inputs), moved_targets)
            target_loss = target_loss * (len(batch_targets) / total_selector_examples)
            target_loss.backward()
            model.zero_grad(set_to_none=True)
        else:
            with torch.no_grad():
                model(moved_inputs)
    selector.remove_hooks()
    model.zero_grad(set_to_none=True)
    return selector


def run_single_selector(
    selector_name: str,
    param_ratio: float,
    args: argparse.Namespace,
    trial_index: int,
    trial_seed: int,
    criterion: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    selector_inputs_cpu: torch.Tensor,
    selector_targets_cpu: torch.Tensor,
    sampled_inputs_cpu: torch.Tensor,
    sampled_targets_cpu: torch.Tensor,
    sampled_inputs_device: torch.Tensor,
    sampled_targets_device: torch.Tensor,
    all_target_count: int,
    retained_inputs_cpu: torch.Tensor,
    retained_targets_cpu: torch.Tensor,
    retained_inputs_device: torch.Tensor,
    retained_targets_device: torch.Tensor,
    eval_target_inputs: torch.Tensor,
    eval_target_targets: torch.Tensor,
    eval_retain_inputs: torch.Tensor,
    eval_retain_targets: torch.Tensor,
    before_metrics: dict[str, float],
    base_state_dict: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, float]:
    model = build_model().to(device)
    model.load_state_dict(base_state_dict)
    model.eval()

    total_loss = build_total_loss(
        model,
        retained_inputs_device,
        retained_targets_device,
        criterion,
        device,
    )
    target_scaling = all_target_count / (len(train_loader.dataset) - all_target_count)
    selector = build_selector(
        selector_name=selector_name,
        model=model,
        ratio=param_ratio,
        selector_inputs=selector_inputs_cpu,
        selector_targets=selector_targets_cpu,
        retained_inputs=retained_inputs_cpu,
        retained_targets=retained_targets_cpu,
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
    target_loss = criterion(model(sampled_inputs_device), sampled_targets_device)
    target_loss = target_loss * target_scaling
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
        current_metrics = evaluate_unlearning_cached(
            model,
            eval_target_inputs,
            eval_target_targets,
            eval_retain_inputs,
            eval_retain_targets,
            criterion,
            device,
            args.eval_batch_size,
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
        selected_metrics = evaluate_unlearning_cached(
            model,
            eval_target_inputs,
            eval_target_targets,
            eval_retain_inputs,
            eval_retain_targets,
            criterion,
            device,
            args.eval_batch_size,
        )

    retain_acc = selected_metrics["retain_acc"]
    self_acc = selected_metrics["self_acc"]
    reported_step = target_step if target_step is not None else args.max_update_steps
    selected_metrics = {
        **selected_metrics,
        "seed": trial_seed,
        "trial": trial_index,
        "selector": selector_name,
        "param_ratio": param_ratio,
        "selected_params": int(len(index_list)),
        "reached_target": reached_target,
        "target_step": target_step,
        "reported_step": reported_step,
        "edit_scale": args.edit_scale,
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
            f"[seed={trial_seed} {selector_name} ratio={param_ratio:.3f}] {status}:",
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
    bundle = create_hf_data_bundle(
        "cifar10",
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        validation=True,
        flatten=False,
        seed=trial_seed,
        dataset_id=args.dataset_id,
    )

    criterion = nn.CrossEntropyLoss()
    all_target_inputs, all_target_targets = collect_target_examples(
        bundle.test_loader, args.target_label
    )
    sampled_inputs, sampled_targets = sample_target_batches(
        all_target_inputs,
        all_target_targets,
        args.batch_size,
        args.num_target_batches,
    )
    retained_inputs, retained_targets = collect_retained_examples(
        bundle.test_loader, args.target_label, 1
    )
    eval_target_inputs, eval_target_targets, eval_retain_inputs, eval_retain_targets = (
        collect_eval_splits(bundle.test_loader, args.target_label)
    )
    sampled_inputs_device = _move_tensor(sampled_inputs, device)
    sampled_targets_device = _move_tensor(sampled_targets, device)
    retained_inputs_device = _move_tensor(retained_inputs, device)
    retained_targets_device = _move_tensor(retained_targets, device)
    if args.cache_test_on_device:
        eval_target_inputs = _move_tensor(eval_target_inputs, device)
        eval_target_targets = _move_tensor(eval_target_targets, device)
        eval_retain_inputs = _move_tensor(eval_retain_inputs, device)
        eval_retain_targets = _move_tensor(eval_retain_targets, device)

    base_model = build_model().to(device)
    load_checkpoint(base_model, args.checkpoint, device)
    base_state = deepcopy(base_model.state_dict())
    before_metrics = evaluate_unlearning_cached(
        base_model,
        eval_target_inputs,
        eval_target_targets,
        eval_retain_inputs,
        eval_retain_targets,
        criterion,
        device,
        args.eval_batch_size,
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
                    train_loader=bundle.train_loader,
                    selector_inputs_cpu=all_target_inputs,
                    selector_targets_cpu=all_target_targets,
                    sampled_inputs_cpu=sampled_inputs,
                    sampled_targets_cpu=sampled_targets,
                    sampled_inputs_device=sampled_inputs_device,
                    sampled_targets_device=sampled_targets_device,
                    all_target_count=len(all_target_targets),
                    retained_inputs_cpu=retained_inputs,
                    retained_targets_cpu=retained_targets,
                    retained_inputs_device=retained_inputs_device,
                    retained_targets_device=retained_targets_device,
                    eval_target_inputs=eval_target_inputs,
                    eval_target_targets=eval_target_targets,
                    eval_retain_inputs=eval_retain_inputs,
                    eval_retain_targets=eval_retain_targets,
                    before_metrics=before_metrics,
                    base_state_dict=base_state,
                    device=device,
                )
            except RuntimeError as error:
                if is_empty_selection_error(error):
                    print(
                        f"[seed={trial_seed} {selector_name} "
                        f"ratio={param_ratio:.3f}] Skipped empty selection: {error}"
                    )
                    continue
                raise
            results.append(metrics)

    ranked = rank_results(results)

    trial_payload: dict[str, object] = {
        "config": {
            "seed": trial_seed,
            "dataset": "cifar10",
            "model": "resnet18",
            "checkpoint": str(args.checkpoint),
            "target_label": args.target_label,
            "selectors": args.selectors,
            "param_ratios": args.param_ratios,
            "tol": args.tol,
            "mu": args.mu,
            "max_iter": args.max_iter,
            "edit_scale": args.edit_scale,
            "max_update_steps": args.max_update_steps,
            "num_target_batches": args.num_target_batches,
            "eval_batch_size": args.eval_batch_size,
            "cache_test_on_device": args.cache_test_on_device,
            "device": args.device,
        },
        "results": ranked,
    }
    trial_path = save_dir / f"seed_{trial_seed:04d}.json"
    trial_path.parent.mkdir(parents=True, exist_ok=True)
    trial_payload, updated_existing = merge_trial_payload(trial_path, trial_payload)
    trial_path.write_text(json.dumps(trial_payload, indent=2))
    action = "Updated existing" if updated_existing else "Saved new"
    print(f"{action} results at {trial_path}")
    return trial_payload


def main() -> None:
    args = parse_args()
    if args.num_trials <= 0:
        raise ValueError("--num-trials must be positive.")

    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    base_bundle = create_hf_data_bundle(
        "cifar10",
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        validation=True,
        flatten=False,
        seed=args.seed,
        dataset_id=args.dataset_id,
    )
    base_model = build_model().to(device)
    load_checkpoint(base_model, args.checkpoint, device)
    base_model.eval()
    base_criterion = nn.CrossEntropyLoss()
    (
        eval_target_inputs,
        eval_target_targets,
        eval_retain_inputs,
        eval_retain_targets,
    ) = collect_eval_splits(base_bundle.test_loader, args.target_label)
    if args.cache_test_on_device:
        eval_target_inputs = _move_tensor(eval_target_inputs, device)
        eval_target_targets = _move_tensor(eval_target_targets, device)
        eval_retain_inputs = _move_tensor(eval_retain_inputs, device)
        eval_retain_targets = _move_tensor(eval_retain_targets, device)
    base_before_metrics = evaluate_unlearning_cached(
        base_model,
        eval_target_inputs,
        eval_target_targets,
        eval_retain_inputs,
        eval_retain_targets,
        base_criterion,
        device,
        args.eval_batch_size,
    )
    print(format_metrics("[global] Before:", base_before_metrics))

    save_root = args.seed_output_dir or EXPERIMENT_ROOT / "results" / "cifar10_resnet18"
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
