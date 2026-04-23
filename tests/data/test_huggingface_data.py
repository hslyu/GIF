from __future__ import annotations

import numpy as np
from datasets import Dataset
from PIL import Image

from gif.data.huggingface import (
    HF_DATASET_SPECS,
    HFImageTorchDataset,
    HFTextTorchDataset,
    build_label_mapping,
    build_text_vocabulary,
    text_collate_fn,
)


def test_hf_image_dataset_returns_tensor_and_label():
    image = Image.fromarray(np.zeros((28, 28), dtype=np.uint8), mode="L")
    dataset = Dataset.from_dict({"image": [image], "label": [3]})
    torch_dataset = HFImageTorchDataset(dataset, HF_DATASET_SPECS["mnist"])

    tensor, label = torch_dataset[0]

    assert tensor.shape == (1, 28, 28)
    assert label == 3


def test_hf_text_dataset_builds_tokenized_examples():
    dataset = Dataset.from_dict(
        {
            "text": ["alpha beta beta", "gamma delta"],
            "label": ["a", "b"],
        }
    )
    vocabulary = build_text_vocabulary(dataset, "text", max_vocab_size=16, min_freq=1)
    label_mapping = build_label_mapping(dataset, "label")
    torch_dataset = HFTextTorchDataset(
        dataset,
        HF_DATASET_SPECS["newsgroup"],
        vocabulary,
        label_mapping,
        max_length=8,
    )

    input_ids, attention_mask, label = torch_dataset[0]

    assert input_ids.ndim == 1
    assert attention_mask.ndim == 1
    assert label == 0


def test_text_collate_fn_pads_batch():
    dataset = Dataset.from_dict(
        {
            "text": ["one two", "three"],
            "label": ["x", "y"],
        }
    )
    vocabulary = build_text_vocabulary(dataset, "text", max_vocab_size=16, min_freq=1)
    label_mapping = build_label_mapping(dataset, "label")
    torch_dataset = HFTextTorchDataset(
        dataset,
        HF_DATASET_SPECS["newsgroup"],
        vocabulary,
        label_mapping,
        max_length=8,
    )

    inputs, labels = text_collate_fn([torch_dataset[0], torch_dataset[1]])
    input_ids, attention_mask = inputs

    assert input_ids.shape == attention_mask.shape
    assert input_ids.shape[0] == 2
    assert labels.shape == (2,)
