#!/usr/bin/env python3
"""Plot the MNIST with FCN selection panel as a standalone figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from plot_selection_results_outlier_filtered_mean import (
    AXIS_LABEL_FONTSIZE,
    COLORS,
    PANEL_LABEL_FONTSIZE,
    aggregate,
    dataset_caption,
    draw_dataset_axis,
    get_retrained_baseline,
    load_retrained_baselines,
    load_seed_rows,
    pretendard_medium,
    pretendard_semibold,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_RETRAINED_RESULTS_DIR = SCRIPT_DIR.parent / "unlearn" / "results"
DEFAULT_OUT_DIR = SCRIPT_DIR
DATASET_CANDIDATES = ("mnist_fcn", "fcn")
EXCLUDED_SELECTORS = {
    "random",
    "highest_k_gradients",
    "highest_k_outputs",
    "lowest_k_gradients",
}
ORIGINAL_IF_LINE_KW = {
    "color": "#666666",
    "linestyle": "-.",
    "linewidth": 1.6,
    "zorder": 4,
}


def load_mnist_rows(results_dir: Path) -> tuple[str, list[dict]]:
    for dataset in DATASET_CANDIDATES:
        dataset_dir = results_dir / dataset
        if not dataset_dir.is_dir():
            continue
        rows = load_seed_rows(dataset_dir)
        if rows:
            return dataset, rows
    candidates = ", ".join(DATASET_CANDIDATES)
    raise SystemExit(f"No MNIST FCN rows found under {results_dir} ({candidates})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--retrained-results-dir",
        type=Path,
        default=DEFAULT_RETRAINED_RESULTS_DIR,
        help="Directory containing JSON files with retrained_baseline metrics.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--metric", default="retain_acc")
    parser.add_argument(
        "--no-smooth-uncertainty",
        action="store_true",
        help="Draw raw mean and interval bands without PCHIP smoothing.",
    )
    parser.add_argument(
        "--legend",
        action="store_true",
        help="Draw the selector legend inside the standalone panel.",
    )
    args = parser.parse_args()

    dataset, rows = load_mnist_rows(args.results_dir)
    retrained_baselines = load_retrained_baselines(
        args.retrained_results_dir, args.metric
    )
    full_summary = aggregate(rows, args.metric)
    random_100_stats = full_summary.get(("random", 1.0))
    random_100_y = None
    if random_100_stats is not None:
        random_100_y = random_100_stats["reached_mean"] or random_100_stats["mean"]
    plot_rows = [row for row in rows if row.get("selector") not in EXCLUDED_SELECTORS]
    COLORS["caps"] = "#3185ff"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4.8))
    handles = draw_dataset_axis(
        ax,
        dataset,
        plot_rows,
        args.metric,
        smooth_uncertainty=not args.no_smooth_uncertainty,
        retrained_baseline=get_retrained_baseline(retrained_baselines, dataset),
        include_legend_labels=args.legend,
    )
    if random_100_y is not None:
        x_min, x_max = ax.get_xlim()
        ax.hlines(random_100_y, x_min, x_max, label="_nolegend_", **ORIGINAL_IF_LINE_KW)
    ax.set_xlabel(
        "Parameter ratio (%)", fontproperties=pretendard_medium(AXIS_LABEL_FONTSIZE)
    )
    ax.set_ylabel(
        "Retain accuracy (%)", fontproperties=pretendard_medium(AXIS_LABEL_FONTSIZE)
    )
    ax.yaxis.set_label_coords(-0.09, 0.5)
    ax.text(
        0.5,
        -0.16,
        dataset_caption(dataset, "a"),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontproperties=pretendard_semibold(PANEL_LABEL_FONTSIZE),
        clip_on=False,
    )

    if args.legend and handles:
        handles.append(
            Line2D([0], [0], label="Original IF (100%)", **ORIGINAL_IF_LINE_KW)
        )
        ax.legend(
            handles,
            [handle.get_label() for handle in handles],
            loc="lower right",
            frameon=True,
            fancybox=True,
            facecolor="white",
            edgecolor="#383838",
            framealpha=0.9,
            prop=pretendard_medium(9),
            handlelength=2.0,
            labelspacing=0.55,
            borderpad=0.5,
        )

    out_path = args.out_dir / "selection_scheme_mnist_fcn.pdf"
    fig.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
    )
    plt.close(fig)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
