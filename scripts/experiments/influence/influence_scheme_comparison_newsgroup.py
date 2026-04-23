#!/usr/bin/env python3
"""Benchmark influence-based model editing schemes on Newsgroup."""

from __future__ import annotations

from _influence_scheme_comparison_text import PROJECT_ROOT, main_for_dataset


def main() -> None:
    main_for_dataset(
        dataset_name="newsgroup",
        result_dir_name="newsgroup",
        default_checkpoint=PROJECT_ROOT
        / "checkpoints"
        / "hf_newsgroup_text_transformer.pth",
        default_target_label=1,
    )


if __name__ == "__main__":
    main()
