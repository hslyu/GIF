#!/usr/bin/env python3
"""Benchmark influence-based model editing schemes on PubMed-RCT20K."""

from __future__ import annotations

from pathlib import Path

from _influence_scheme_comparison_text import PROJECT_ROOT, main_for_dataset


def main() -> None:
    main_for_dataset(
        dataset_name="pubmed_rct20k",
        result_dir_name="pubmed_rct20k",
        default_checkpoint=PROJECT_ROOT
        / "checkpoints"
        / "hf_pubmed_rct20k_hf_text_encoder.pth",
        default_target_label=1,
        default_model="hf_text_encoder",
        default_pretrained_text_model_name="google/bert_uncased_L-2_H-128_A-2",
    )


if __name__ == "__main__":
    main()
