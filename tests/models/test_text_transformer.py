from __future__ import annotations

import torch

from gif.models import TextTransformerClassifier


def test_text_transformer_forward_shape():
    model = TextTransformerClassifier(
        vocab_size=64,
        max_len=16,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        num_classes=5,
    )
    input_ids = torch.randint(0, 20, (3, 10))
    attention_mask = torch.ones_like(input_ids)

    logits = model(input_ids, attention_mask)

    assert logits.shape == (3, 5)
