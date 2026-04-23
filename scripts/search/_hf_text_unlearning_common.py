from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import pad

from gif.data.huggingface import create_hf_data_bundle
from gif.influence import (
    FreezingInfluence,
    HyperInfluence,
    InfluenceFunction,
    SecondOrderInfluence,
    TracIn,
    generalized_influence,
    project_subset,
)
from gif.selection import CAPS

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["net"])


class TextBatchDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (self.input_ids[index], self.attention_mask[index]), self.labels[index]


def build_loader(input_ids, attention_mask, labels, batch_size):
    return DataLoader(
        TextBatchDataset(input_ids, attention_mask, labels),
        batch_size=min(batch_size, len(labels)),
        shuffle=False,
    )


def collect_examples(dataloader):
    ids_list, mask_list, labels_list = [], [], []
    for (input_ids, attention_mask), labels in dataloader:
        ids_list.append(input_ids)
        mask_list.append(attention_mask)
        labels_list.append(labels)

    if not ids_list:
        raise RuntimeError("No text examples were collected from the dataloader.")

    max_len = max(tensor.shape[1] for tensor in ids_list)
    padded_ids = []
    padded_masks = []
    for input_ids, attention_mask in zip(ids_list, mask_list):
        pad_width = max_len - input_ids.shape[1]
        if pad_width > 0:
            input_ids = pad(input_ids, (0, pad_width), value=0)
            attention_mask = pad(attention_mask, (0, pad_width), value=0)
        padded_ids.append(input_ids)
        padded_masks.append(attention_mask)

    return (
        torch.cat(padded_ids, dim=0),
        torch.cat(padded_masks, dim=0),
        torch.cat(labels_list, dim=0),
    )


def collect_target_examples(dataloader, target_label):
    ids, masks, labels = collect_examples(dataloader)
    mask = labels == target_label
    if not torch.any(mask):
        raise RuntimeError(f"No examples found for target label {target_label}.")
    return ids[mask], masks[mask], labels[mask]


def collect_retained_examples(dataloader, target_label, num_batches):
    ids_list, masks_list, labels_list = [], [], []
    used = 0
    for (input_ids, attention_mask), labels in dataloader:
        if used >= num_batches:
            break
        mask = labels != target_label
        if not torch.any(mask):
            continue
        ids_list.append(input_ids[mask])
        masks_list.append(attention_mask[mask])
        labels_list.append(labels[mask])
        used += 1
    if not ids_list:
        raise RuntimeError("No retained examples found.")
    return (
        torch.cat(ids_list, dim=0),
        torch.cat(masks_list, dim=0),
        torch.cat(labels_list, dim=0),
    )


def sample_target_batch(input_ids, attention_mask, labels, num_samples):
    num_samples = min(num_samples, len(labels))
    indices = np.random.choice(len(labels), size=num_samples, replace=False)
    return input_ids[indices], attention_mask[indices], labels[indices]


def evaluate_split(model, dataloader, criterion, label, include_label, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for (input_ids, attention_mask), targets in dataloader:
            mask = targets == label if include_label else targets != label
            if not torch.any(mask):
                continue
            input_ids = input_ids[mask].to(device)
            attention_mask = attention_mask[mask].to(device)
            targets = targets[mask].to(device)
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * targets.size(0)
            total_correct += outputs.argmax(dim=1).eq(targets).sum().item()
            total_examples += targets.size(0)
    if total_examples == 0:
        raise RuntimeError("Split evaluation received zero examples.")
    return total_loss / total_examples, 100.0 * total_correct / total_examples


def f1_unlearning_score(self_acc, retain_acc):
    self_acc /= 100.0
    retain_acc /= 100.0
    if self_acc == 1.0 and retain_acc == 0.0:
        return 0.0
    return 2.0 * (1.0 - self_acc) * retain_acc / (1.0 - self_acc + retain_acc)


def evaluate_unlearning(model, dataloader, criterion, target_label, device):
    self_loss, self_acc = evaluate_split(
        model, dataloader, criterion, target_label, True, device
    )
    retain_loss, retain_acc = evaluate_split(
        model, dataloader, criterion, target_label, False, device
    )
    return {
        "self_loss": self_loss,
        "self_acc": self_acc,
        "retain_loss": retain_loss,
        "retain_acc": retain_acc,
        "score": f1_unlearning_score(self_acc, retain_acc),
    }


def build_total_loss(model, retained_ids, retained_masks, retained_targets, criterion, device):
    return criterion(
        model(retained_ids.to(device), retained_masks.to(device)),
        retained_targets.to(device),
    )


def select_parameters(
    model,
    criterion,
    sampled_ids,
    sampled_masks,
    sampled_targets,
    retained_ids,
    retained_masks,
    retained_targets,
    param_ratio,
    caps_lam,
    batch_size,
    device,
):
    selector = CAPS(model, ratio=param_ratio, lam=caps_lam)
    target_loader = build_loader(sampled_ids, sampled_masks, sampled_targets, batch_size)
    retained_loader = build_loader(retained_ids, retained_masks, retained_targets, batch_size)
    selector.fit(
        target_loader=target_loader,
        retained_loader=retained_loader,
        criterion=criterion,
        device=device,
    )
    return selector


def compute_method_update(
    scheme,
    model,
    train_loader,
    criterion,
    sampled_ids,
    sampled_masks,
    sampled_targets,
    all_target_count,
    retained_ids,
    retained_masks,
    retained_targets,
    param_ratio,
    caps_lam,
    batch_size,
    tol,
    mu,
    max_iter,
    device,
    trajectory_dir=None,
    hyperinf_beta_scale=0.9,
):
    total_loss = build_total_loss(
        model, retained_ids, retained_masks, retained_targets, criterion, device
    )
    target_scaling = all_target_count / (len(train_loader.dataset) - all_target_count)
    selector = select_parameters(
        model,
        criterion,
        sampled_ids,
        sampled_masks,
        sampled_targets,
        retained_ids,
        retained_masks,
        retained_targets,
        param_ratio,
        caps_lam,
        batch_size,
        device,
    )
    index_list = selector.get_parameters()
    target_loss = criterion(
        model(sampled_ids.to(device), sampled_masks.to(device)),
        sampled_targets.to(device),
    ) * target_scaling

    if scheme == "gif":
        influence = generalized_influence(
            model, total_loss, target_loss, index_list, mu=mu, tol=tol, max_iter=max_iter
        )
    elif scheme == "influence":
        influence = project_subset(
            InfluenceFunction().compute(
                model=model,
                total_loss=total_loss,
                loss=target_loss,
                mu=mu,
                tol=tol,
                max_iter=max_iter,
            ),
            index_list,
        )
    elif scheme == "second_influence":
        influence = project_subset(
            SecondOrderInfluence().compute(
                model=model,
                total_loss=total_loss,
                target_loss=target_loss,
                num_total_data=len(train_loader.dataset),
                num_target_data=all_target_count,
                tol=tol,
                step=0.5,
                max_iter=max_iter,
                normalizer=1.0,
            ),
            index_list,
        )
    elif scheme == "freeze_influence":
        influence = FreezingInfluence().compute(
            model=model,
            total_loss=total_loss,
            target_loss=target_loss,
            index_list=index_list,
            tol=tol,
            step=0.5,
            max_iter=max_iter,
            normalizer=1.0,
        )
    elif scheme == "tracin":
        if trajectory_dir is None:
            raise RuntimeError("TracIn requires a trajectory_dir.")
        influence = TracIn().compute_update(
            model=model,
            trajectory_dir=trajectory_dir,
            target_inputs=(sampled_ids.to(device), sampled_masks.to(device)),
            target_targets=sampled_targets.to(device),
            criterion=criterion,
            device=device,
            index_list=index_list,
        ) * target_scaling
    elif scheme == "hyperinf":
        influence = HyperInfluence().compute(
            model=model,
            total_loss=total_loss,
            target_loss=target_loss,
            index_list=index_list,
            beta_scale=hyperinf_beta_scale,
            tol=tol,
            max_iter=max_iter,
        )
    else:
        raise ValueError(f"Unsupported scheme: {scheme}")

    norm = torch.norm(influence)
    if torch.isnan(norm) or norm.item() == 0.0:
        raise RuntimeError(f"{scheme} update has zero or NaN norm.")
    return selector, influence / norm


def format_metrics(prefix, metrics):
    return (
        f"{prefix} retain_acc={metrics['retain_acc']:.2f}% "
        f"retain_loss={metrics['retain_loss']:.4f} "
        f"self_acc={metrics['self_acc']:.2f}% "
        f"self_loss={metrics['self_loss']:.4f} "
        f"score={metrics['score']:.4f}"
    )


def run_experiment(args, model_factory):
    device = torch.device(args.device)
    set_seed(args.seed)
    bundle = create_hf_data_bundle(
        args.dataset,
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
    criterion = nn.CrossEntropyLoss()
    all_target_ids, all_target_masks, all_target_targets = collect_target_examples(
        bundle.test_loader, args.target_label
    )
    sampled_ids, sampled_masks, sampled_targets = sample_target_batch(
        all_target_ids, all_target_masks, all_target_targets, args.num_target_samples
    )
    retained_ids, retained_masks, retained_targets = collect_retained_examples(
        bundle.test_loader, args.target_label, args.num_retain_batches
    )

    base_model = model_factory(bundle).to(device)
    load_checkpoint(base_model, args.checkpoint, device)
    base_state = deepcopy(base_model.state_dict())

    results = {}
    for scheme in args.schemes:
        model = model_factory(bundle).to(device)
        model.load_state_dict(base_state)
        before = evaluate_unlearning(model, bundle.test_loader, criterion, args.target_label, device)
        print(format_metrics(f"[{scheme}] Before:", before))
        selector, normalized_update = compute_method_update(
            scheme,
            model,
            bundle.train_loader,
            criterion,
            sampled_ids,
            sampled_masks,
            sampled_targets,
            len(all_target_targets),
            retained_ids,
            retained_masks,
            retained_targets,
            args.param_ratio,
            args.caps_lam,
            args.batch_size,
            args.tol,
            args.mu,
            args.max_iter,
            device,
            trajectory_dir=args.trajectory_dir,
            hyperinf_beta_scale=args.hyperinf_beta_scale,
        )
        selector.update_network(normalized_update * args.edit_scale)
        after = evaluate_unlearning(model, bundle.test_loader, criterion, args.target_label, device)
        print(format_metrics(f"[{scheme}] Step 1:", after))
        results[scheme] = after
    return results
