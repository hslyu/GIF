#!/usr/bin/env python3
"""Compare parameter selection schemes on Newsgroup unlearning."""

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

from _hf_text_unlearning_common import (  # noqa: E402
    build_loader,
    build_total_loss,
    collect_retained_examples,
    collect_target_examples,
    evaluate_unlearning,
    format_metrics,
    load_checkpoint,
    set_seed,
)

from gif.data.huggingface import create_hf_data_bundle  # noqa: E402
from gif.influence import generalized_influence  # noqa: E402
from gif.models import (  # noqa: E402
    PretrainedTextEncoderClassifier,
    TextTransformerClassifier,
)
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

GRADIENT_SELECTOR_NAMES = {"highest_k_gradients", "lowest_k_gradients"}
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
    "caps",
    "reverse_caps",
    "highest_k_outputs",
    "highest_k_gradients",
    "lowest_k_outputs",
    "lowest_k_gradients",
    "random",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare selection schemes on Newsgroup with a fixed generalized-influence update."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "hf_newsgroup_hf_text_encoder.pth",
    )
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument(
        "--model",
        choices=["text_transformer", "hf_text_encoder"],
        default="hf_text_encoder",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--num-target-batches", type=int, default=10)
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
    parser.add_argument("--caps-lam", type=float, default=1e-5)
    parser.add_argument("--dataset-id", type=str, default=None)
    parser.add_argument("--max-text-length", type=int, default=256)
    parser.add_argument("--max-vocab-size", type=int, default=30000)
    parser.add_argument("--min-token-freq", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--text-num-layers", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=512)
    parser.add_argument(
        "--pretrained-text-model-name",
        type=str,
        default="google/bert_uncased_L-2_H-128_A-2",
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
        / "newsgroup"
        / f"seed_{args.seed:04d}_trials_{args.num_trials:03d}.json"
    )


def build_model(bundle, args: argparse.Namespace) -> nn.Module:
    if args.model == "hf_text_encoder":
        return PretrainedTextEncoderClassifier(
            pretrained_model_name=args.pretrained_text_model_name,
            num_classes=bundle.num_classes,
        )
    return TextTransformerClassifier(
        vocab_size=min(bundle.vocabulary.size, args.max_vocab_size),
        max_len=args.max_text_length,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.text_num_layers,
        dim_feedforward=args.dim_feedforward,
        num_classes=bundle.num_classes,
    )


def is_empty_selection_error(error: RuntimeError) -> bool:
    message = str(error)
    return any(marker in message for marker in EMPTY_SELECTION_ERROR_MARKERS)


def build_selector(
    selector_name: str,
    model: nn.Module,
    ratio: float,
    selector_ids: torch.Tensor,
    selector_masks: torch.Tensor,
    selector_targets: torch.Tensor,
    retained_ids: torch.Tensor,
    retained_masks: torch.Tensor,
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
        target_loader = build_loader(
            selector_ids, selector_masks, selector_targets, args.batch_size
        )
        retained_loader = build_loader(
            retained_ids, retained_masks, retained_targets, args.batch_size
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
    selector_loader = build_loader(
        selector_ids, selector_masks, selector_targets, args.batch_size
    )
    total_selector_examples = len(selector_targets)
    for (batch_ids, batch_masks), batch_targets in selector_loader:
        moved_ids = batch_ids.to(device)
        moved_masks = batch_masks.to(device)
        moved_targets = batch_targets.to(device)
        if selector_name in GRADIENT_SELECTOR_NAMES:
            target_loss = criterion(model(moved_ids, moved_masks), moved_targets)
            target_loss = target_loss * (len(batch_targets) / total_selector_examples)
            target_loss.backward()
            model.zero_grad(set_to_none=True)
        else:
            with torch.no_grad():
                model(moved_ids, moved_masks)
    selector.remove_hooks()
    model.zero_grad(set_to_none=True)
    return selector


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


def run_single_selector(
    selector_name: str,
    param_ratio: float,
    args: argparse.Namespace,
    trial_index: int,
    trial_seed: int,
    criterion: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    selector_ids: torch.Tensor,
    selector_masks: torch.Tensor,
    selector_targets: torch.Tensor,
    sampled_ids: torch.Tensor,
    sampled_masks: torch.Tensor,
    sampled_targets: torch.Tensor,
    all_target_count: int,
    retained_ids: torch.Tensor,
    retained_masks: torch.Tensor,
    retained_targets: torch.Tensor,
    before_metrics: dict[str, float],
    base_state_dict: dict[str, torch.Tensor],
    model_factory,
    device: torch.device,
) -> dict[str, float]:
    model = model_factory().to(device)
    model.load_state_dict(base_state_dict)
    model.eval()

    total_loss = build_total_loss(
        model,
        retained_ids,
        retained_masks,
        retained_targets,
        criterion,
        device,
    )
    target_scaling = all_target_count / (len(train_loader.dataset) - all_target_count)
    selector = build_selector(
        selector_name=selector_name,
        model=model,
        ratio=param_ratio,
        selector_ids=selector_ids,
        selector_masks=selector_masks,
        selector_targets=selector_targets,
        retained_ids=retained_ids,
        retained_masks=retained_masks,
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
        criterion(
            model(sampled_ids.to(device), sampled_masks.to(device)),
            sampled_targets.to(device),
        )
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

    target_metrics: dict[str, float] | None = None
    target_step = None
    final_metrics = None
    for step in range(1, args.max_update_steps + 1):
        selector.update_network(normalized_update * args.edit_scale)
        final_metrics = evaluate_unlearning(
            model, test_loader, criterion, args.target_label, device
        )
        if final_metrics["self_acc"] <= args.target_self_acc:
            target_step = step
            target_metrics = deepcopy(final_metrics)
            break

    reached_target = target_step is not None
    selected_metrics = target_metrics if target_metrics is not None else final_metrics
    if selected_metrics is None:
        selected_metrics = evaluate_unlearning(
            model, test_loader, criterion, args.target_label, device
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
        "newsgroup",
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        validation=True,
        seed=trial_seed,
        dataset_id=args.dataset_id,
        max_text_length=args.max_text_length,
        max_vocab_size=args.max_vocab_size,
        min_token_freq=args.min_token_freq,
        pretrained_text_model_name=(
            args.pretrained_text_model_name if args.model == "hf_text_encoder" else None
        ),
    )

    criterion = nn.CrossEntropyLoss()
    all_target_ids, all_target_masks, all_target_targets = collect_target_examples(
        bundle.test_loader, args.target_label
    )
    sampled_ids, sampled_masks, sampled_targets = sample_target_batches(
        all_target_ids,
        all_target_masks,
        all_target_targets,
        args.batch_size,
        args.num_target_batches,
    )
    retained_ids, retained_masks, retained_targets = collect_retained_examples(
        bundle.test_loader, args.target_label, 1
    )

    base_model = build_model(bundle, args).to(device)
    load_checkpoint(base_model, args.checkpoint, device)
    base_model.eval()
    base_state = deepcopy(base_model.state_dict())
    before_metrics = evaluate_unlearning(
        base_model, bundle.test_loader, criterion, args.target_label, device
    )
    model_factory = lambda: build_model(bundle, args)

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
                    test_loader=bundle.test_loader,
                    selector_ids=all_target_ids,
                    selector_masks=all_target_masks,
                    selector_targets=all_target_targets,
                    sampled_ids=sampled_ids,
                    sampled_masks=sampled_masks,
                    sampled_targets=sampled_targets,
                    all_target_count=len(all_target_targets),
                    retained_ids=retained_ids,
                    retained_masks=retained_masks,
                    retained_targets=retained_targets,
                    before_metrics=before_metrics,
                    base_state_dict=base_state,
                    model_factory=model_factory,
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

    ranked = sorted(
        results,
        key=lambda item: (
            item["reached_target"],
            item["retain_acc"],
            -item["target_step"] if item["target_step"] is not None else float("-inf"),
        ),
        reverse=True,
    )

    trial_payload: dict[str, object] = {
        "config": {
            "seed": trial_seed,
            "dataset": "newsgroup",
            "model": args.model,
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
            "device": args.device,
            "pretrained_text_model_name": (
                args.pretrained_text_model_name
                if args.model == "hf_text_encoder"
                else None
            ),
        },
        "results": ranked,
    }
    trial_path = save_dir / f"seed_{trial_seed:04d}.json"
    trial_path.parent.mkdir(parents=True, exist_ok=True)
    trial_path.write_text(json.dumps(trial_payload, indent=2))
    print(f"Saved results to {trial_path}")
    return trial_payload


def main() -> None:
    args = parse_args()
    if args.num_trials <= 0:
        raise ValueError("--num-trials must be positive.")

    set_seed(args.seed)
    device = torch.device(args.device)
    base_bundle = create_hf_data_bundle(
        "newsgroup",
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        validation=True,
        seed=args.seed,
        dataset_id=args.dataset_id,
        max_text_length=args.max_text_length,
        max_vocab_size=args.max_vocab_size,
        min_token_freq=args.min_token_freq,
        pretrained_text_model_name=(
            args.pretrained_text_model_name if args.model == "hf_text_encoder" else None
        ),
    )
    base_model = build_model(base_bundle, args).to(device)
    load_checkpoint(base_model, args.checkpoint, device)
    base_model.eval()
    base_criterion = nn.CrossEntropyLoss()
    base_before_metrics = evaluate_unlearning(
        base_model, base_bundle.test_loader, base_criterion, args.target_label, device
    )
    print(format_metrics("[global] Before:", base_before_metrics))

    save_root = args.seed_output_dir or EXPERIMENT_ROOT / "results" / "newsgroup"
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
