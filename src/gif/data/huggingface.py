from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms


@dataclass(frozen=True)
class HFDatasetSpec:
    name: str
    dataset_id: str
    task_type: str
    label_key: str
    train_split: str
    test_split: str
    validation_split: str | None = None
    dataset_config: str | None = None
    image_key: str | None = None
    text_key: str | None = None
    num_classes: int | None = None
    in_channels: int | None = None
    image_size: int | None = None
    trust_remote_code: bool = False


HF_DATASET_SPECS = {
    "mnist": HFDatasetSpec(
        name="mnist",
        dataset_id="ylecun/mnist",
        task_type="image",
        image_key="image",
        label_key="label",
        train_split="train",
        test_split="test",
        num_classes=10,
        in_channels=1,
        image_size=28,
    ),
    "cifar10": HFDatasetSpec(
        name="cifar10",
        dataset_id="tanganke/cifar10",
        task_type="image",
        image_key="image",
        label_key="label",
        train_split="train",
        test_split="test",
        num_classes=10,
        in_channels=3,
        image_size=32,
    ),
    "svhn": HFDatasetSpec(
        name="svhn",
        dataset_id="svhn",
        dataset_config="cropped_digits",
        task_type="image",
        image_key="image",
        label_key="label",
        train_split="train",
        test_split="test",
        num_classes=10,
        in_channels=3,
        image_size=32,
    ),
    "newsgroup": HFDatasetSpec(
        name="newsgroup",
        dataset_id="SetFit/20_newsgroups",
        task_type="text",
        text_key="text",
        label_key="label",
        train_split="train",
        test_split="test",
        num_classes=20,
    ),
    "pubmed_rct20k": HFDatasetSpec(
        name="pubmed_rct20k",
        dataset_id="armanc/pubmed-rct20k",
        task_type="text",
        text_key="text",
        label_key="label",
        train_split="train",
        test_split="test",
        validation_split="validation",
    ),
}

IMAGE_NORMALIZATION = {
    "mnist": ((0.1307,), (0.3081,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "svhn": ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
}


def get_hf_dataset_spec(name: str) -> HFDatasetSpec:
    try:
        return HF_DATASET_SPECS[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported Hugging Face dataset: {name}") from exc


def load_hf_dataset_splits(
    dataset_name: str,
    *,
    cache_dir: Path,
    validation: bool = True,
    validation_ratio: float = 0.1,
    seed: int = 0,
    dataset_id: str | None = None,
) -> tuple[HFDatasetSpec, Dataset, Dataset | None, Dataset]:
    spec = get_hf_dataset_spec(dataset_name)
    resolved_dataset_id = dataset_id or spec.dataset_id
    load_kwargs = {
        "cache_dir": str(cache_dir),
        "trust_remote_code": spec.trust_remote_code,
    }
    if spec.dataset_config is not None:
        dataset = load_dataset(
            resolved_dataset_id,
            spec.dataset_config,
            **load_kwargs,
        )
    else:
        dataset = load_dataset(
            resolved_dataset_id,
            **load_kwargs,
        )
    if not isinstance(dataset, DatasetDict):
        raise RuntimeError(f"Expected DatasetDict for {resolved_dataset_id}.")

    train_dataset = dataset[spec.train_split]
    test_dataset = dataset[spec.test_split]

    if validation:
        if spec.validation_split is not None and spec.validation_split in dataset:
            val_dataset = dataset[spec.validation_split]
        else:
            split = train_dataset.train_test_split(
                test_size=validation_ratio,
                seed=seed,
            )
            train_dataset = split["train"]
            val_dataset = split["test"]
    else:
        val_dataset = None

    return spec, train_dataset, val_dataset, test_dataset


class HFImageTorchDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: Dataset, spec: HFDatasetSpec, flatten: bool = False):
        self.dataset = dataset
        self.spec = spec
        mean, std = IMAGE_NORMALIZATION[spec.name]
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                transforms.Lambda(lambda x: x.view(-1) if flatten else x),
            ]
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def _extract_label(self, raw_label) -> int:
        if isinstance(raw_label, (int, np.integer)):
            return int(raw_label)
        raise TypeError(
            f"Unsupported label type for image dataset {self.spec.name}: "
            f"{type(raw_label).__name__}. Expected a scalar class index."
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        item = self.dataset[index]
        image = item[self.spec.image_key]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))
        tensor = self.transform(image)
        label = self._extract_label(item[self.spec.label_key])
        return tensor, label


def basic_tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


@dataclass
class TextVocabulary:
    stoi: dict[str, int]
    itos: list[str]
    pad_idx: int = 0
    unk_idx: int = 1

    def encode(self, text: str, max_length: int) -> list[int]:
        tokens = basic_tokenize(text)[:max_length]
        return [self.stoi.get(token, self.unk_idx) for token in tokens]

    @property
    def size(self) -> int:
        return len(self.itos)


def build_text_vocabulary(
    dataset: Dataset,
    text_key: str,
    *,
    max_vocab_size: int = 30000,
    min_freq: int = 2,
) -> TextVocabulary:
    counts: dict[str, int] = {}
    for item in dataset:
        for token in basic_tokenize(item[text_key]):
            counts[token] = counts.get(token, 0) + 1

    sorted_tokens = sorted(
        (token for token, count in counts.items() if count >= min_freq),
        key=lambda token: (-counts[token], token),
    )[: max_vocab_size - 2]

    itos = ["<pad>", "<unk>", *sorted_tokens]
    stoi = {token: idx for idx, token in enumerate(itos)}
    return TextVocabulary(stoi=stoi, itos=itos)


def build_label_mapping(dataset: Dataset, label_key: str) -> dict[object, int]:
    labels = sorted({item[label_key] for item in dataset})
    return {label: idx for idx, label in enumerate(labels)}


class HFTextTorchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: Dataset,
        spec: HFDatasetSpec,
        vocabulary: TextVocabulary,
        label_mapping: dict[object, int],
        *,
        max_length: int = 256,
    ):
        self.dataset = dataset
        self.spec = spec
        self.vocabulary = vocabulary
        self.label_mapping = label_mapping
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        item = self.dataset[index]
        ids = self.vocabulary.encode(item[self.spec.text_key], self.max_length)
        if len(ids) == 0:
            ids = [self.vocabulary.unk_idx]
        input_ids = torch.tensor(ids, dtype=torch.long)
        attention_mask = torch.ones(len(ids), dtype=torch.long)
        label = self.label_mapping[item[self.spec.label_key]]
        return input_ids, attention_mask, label


def text_collate_fn(batch):
    input_ids, attention_masks, labels = zip(*batch)
    batch_size = len(input_ids)
    max_len = max(max(ids.numel(), 1) for ids in input_ids)
    padded_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    padded_mask = torch.zeros(batch_size, max_len, dtype=torch.long)

    for row_idx, (ids, mask) in enumerate(zip(input_ids, attention_masks)):
        if ids.numel() == 0:
            continue
        padded_ids[row_idx, : ids.numel()] = ids
        padded_mask[row_idx, : mask.numel()] = mask

    return (padded_ids, padded_mask), torch.tensor(labels, dtype=torch.long)


@dataclass
class HFDataBundle:
    spec: HFDatasetSpec
    train_loader: DataLoader
    val_loader: DataLoader | None
    test_loader: DataLoader
    flatten: bool
    num_classes: int
    in_channels: int | None = None
    image_size: int | None = None
    vocabulary: TextVocabulary | None = None


def create_hf_data_bundle(
    dataset_name: str,
    *,
    data_root: Path,
    batch_size: int,
    num_workers: int,
    validation: bool = True,
    flatten: bool = False,
    seed: int = 0,
    dataset_id: str | None = None,
    max_text_length: int = 256,
    max_vocab_size: int = 30000,
    min_token_freq: int = 2,
) -> HFDataBundle:
    spec, train_split, val_split, test_split = load_hf_dataset_splits(
        dataset_name,
        cache_dir=data_root / "huggingface",
        validation=validation,
        seed=seed,
        dataset_id=dataset_id,
    )

    if spec.task_type == "image":
        train_dataset = HFImageTorchDataset(train_split, spec, flatten=flatten)
        val_dataset = (
            None if val_split is None else HFImageTorchDataset(val_split, spec, flatten=flatten)
        )
        test_dataset = HFImageTorchDataset(test_split, spec, flatten=flatten)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = (
            None
            if val_dataset is None
            else DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        return HFDataBundle(
            spec=spec,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            flatten=flatten,
            num_classes=spec.num_classes,
            in_channels=spec.in_channels,
            image_size=spec.image_size,
        )

    vocabulary = build_text_vocabulary(
        train_split,
        spec.text_key,
        max_vocab_size=max_vocab_size,
        min_freq=min_token_freq,
    )
    label_mapping = build_label_mapping(train_split, spec.label_key)
    train_dataset = HFTextTorchDataset(
        train_split,
        spec,
        vocabulary,
        label_mapping,
        max_length=max_text_length,
    )
    val_dataset = (
        None
        if val_split is None
        else HFTextTorchDataset(
            val_split,
            spec,
            vocabulary,
            label_mapping,
            max_length=max_text_length,
        )
    )
    test_dataset = HFTextTorchDataset(
        test_split,
        spec,
        vocabulary,
        label_mapping,
        max_length=max_text_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=text_collate_fn,
    )
    val_loader = (
        None
        if val_dataset is None
        else DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=text_collate_fn,
        )
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=text_collate_fn,
    )
    return HFDataBundle(
        spec=spec,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        flatten=False,
        num_classes=len(label_mapping),
        vocabulary=vocabulary,
    )
