from __future__ import annotations

import torch
import torch.nn as nn


class TextClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_classes: int,
        *,
        pad_idx: int = 0,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = input_ids.ne(self.pad_idx)

        embedded = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1).to(embedded.dtype)
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.classifier(pooled)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout_prob: float,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_prob)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(dim_feedforward, d_model),
        )

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        residual = x
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        key_mask = attention_mask.unsqueeze(1).unsqueeze(2).bool()
        scores = scores.masked_fill(~key_mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        context = torch.matmul(self.dropout(attn), v)
        x = self._merge_heads(context)
        x = self.out_proj(x)
        x = self.norm1(residual + self.dropout(x))

        residual = x
        x = self.ffn(x)
        x = self.norm2(residual + self.dropout(x))
        return x


class TextTransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        num_classes: int,
        *,
        pad_idx: int = 0,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.max_len = max_len
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.position_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout_prob=dropout_prob,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = input_ids.ne(self.pad_idx)

        x = self.token_embedding(input_ids)
        x = self.position_encoding(x)
        for layer in self.layers:
            x = layer(x, attention_mask)
        mask = attention_mask.unsqueeze(-1).to(x.dtype)
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


def _require_transformers():
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise ImportError(
            "transformers is required for pretrained text encoder models. "
            "Install the package with `pip install transformers`."
        ) from exc
    return AutoModel


class PretrainedTextEncoderClassifier(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str,
        num_classes: int,
        *,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        auto_model = _require_transformers()
        self.pretrained_model_name = pretrained_model_name
        self.encoder = auto_model.from_pretrained(pretrained_model_name)
        hidden_size = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        token_features = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(token_features.dtype)
        pooled = (token_features * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)
