from __future__ import annotations

import torch

from gif.models import TextClassifier


def test_text_classifier_forward_shape():
    model = TextClassifier(
        vocab_size=32,
        embed_dim=16,
        hidden_dim=8,
        num_classes=5,
    )
    input_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.long)

    logits = model(input_ids, attention_mask)

    assert logits.shape == (2, 5)


def test_text_classifier_ignores_padding_via_mask():
    torch.manual_seed(0)
    model = TextClassifier(
        vocab_size=16,
        embed_dim=8,
        hidden_dim=4,
        num_classes=3,
    )
    input_ids = torch.tensor([[1, 2, 0, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)

    logits_masked = model(input_ids, attention_mask)
    logits_unmasked = model(input_ids, torch.ones_like(attention_mask))

    assert not torch.allclose(logits_masked, logits_unmasked)
