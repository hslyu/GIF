#!/usr/bin/env python3
"""
Plot selection scheme comparison results.

The experiment runners write one JSON file per seed under results/<dataset>/.
This script aggregates those seed files and plots each selector's performance
over the parameter-ratio sweep as mean +/- std.
"""

from __future__ import annotations

import argparse
import colorsys
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import PchipInterpolator

SELECTOR_ORDER = [
    "random",
    "caps",
    "highest_k_gradients",
    "highest_k_outputs",
    "reverse_caps",
    "lowest_k_gradients",
    "lowest_k_outputs",
]

LEGEND_ORDER = [
    "random",
    "caps",
    "highest_k_gradients",
    "highest_k_outputs",
    "reverse_caps",
    "lowest_k_gradients",
    "lowest_k_outputs",
]
RETRAINED_LEGEND_LABEL = "From-scratch retrain"

SELECTOR_LABELS = {
    "caps": "CAPS",
    "reverse_caps": "Reverse caps",
    "highest_k_outputs": "Highest outputs",
    "highest_k_gradients": "Highest gradients",
    "lowest_k_outputs": "Lowest outputs",
    "lowest_k_gradients": "Lowest gradients",
    "random": "Random",
}

SELECTOR_LINE_ZORDER = {
    selector: 100 - idx for idx, selector in enumerate(SELECTOR_ORDER)
}
SELECTOR_BAND_ZORDER = {
    selector: 10 - idx * 0.1 for idx, selector in enumerate(SELECTOR_ORDER)
}

COLORS = {
    "caps": "#073b8e",
    "highest_k_gradients": "#2575f4",
    "highest_k_outputs": "#5cbcbf",
    "lowest_k_outputs": "#f4aa62",
    "lowest_k_gradients": "#e8563d",
    "reverse_caps": "#b21424",
    "random": "#666666",
}

MARKERS = {
    "caps": "o",
    "highest_k_gradients": "8",
    "highest_k_outputs": "h",
    "random": "s",
    "reverse_caps": "^",
    "lowest_k_outputs": "v",
    "lowest_k_gradients": "D",
}

MARKER_SIZES = {
    "caps": 7.6,
    "highest_k_gradients": 7.6,
    "highest_k_outputs": 7.6,
    "random": 6.6,
    "reverse_caps": 7.6,
    "lowest_k_outputs": 7.6,
    "lowest_k_gradients": 6.6,
}
INSET_MARKER_SIZES = {selector: size - 1.3 for selector, size in MARKER_SIZES.items()}

GRID_KW = {"linestyle": (0, (5, 5)), "linewidth": 0.5, "color": "#e0e0e0"}
RETRAINED_LINE_KW = {
    "color": "#222222",
    "linestyle": (0, (2, 2)),
    "linewidth": 1.4,
    "zorder": 2,
}
MAIN_XTICKS = [0.05, 0.2, 0.4, 0.6, 0.8, 1.0]
MAIN_XTICK_LABELS = ["5", "20", "40", "60", "80", "100"]
MAIN_XLIM = (0.03, 1.02)
PCHIP_SAMPLES_PER_INTERVAL = 24

DATASETS = [
    ("mnist_fcn", "MNIST", "FCN"),
    ("cifar10_resnet18", "CIFAR-10", "ResNet-18"),
    ("svhn_vgg11", "SVHN", "VGG-11"),
    ("newsgroup", "Newsgroups", "BERT-L2-H128"),
    ("pubmed_rct20k", "PubMed 20k RCT", "BERT-L2-H128"),
    ("fcn", "MNIST", "FCN"),
]
DATASET_ORDER = [dataset_key for dataset_key, _, _ in DATASETS]
DATASET_TITLES = {
    dataset_key: f"{dataset_label}\n{model_label}"
    for dataset_key, dataset_label, model_label in DATASETS
}
DATASET_CAPTIONS = {
    "mnist_fcn": "MNIST with FCN",
    "fcn": "MNIST with FCN",
    "cifar10_resnet18": "CIFAR-10 with ResNet-18",
    "svhn_vgg11": "SVHN with VGG-11",
    "newsgroup": "20 Newsgroups with BERT-L2-H128",
    "pubmed_rct20k": "PubMed RCT with BERT-L2-H128",
}
DATASET_ALIASES = {
    "mnist_fcn": ["mnist_fcn", "fcn"],
    "fcn": ["fcn", "mnist_fcn"],
}

DATASET_YLIMS = {
    "mnist_fcn": (20.0, 100.0),
    "fcn": (20.0, 100.0),
    "cifar10_resnet18": (60, 90),
    "newsgroup": (15, 70),
    "pubmed_rct20k": (60.0, 88.0),
}
DATASET_YTICKS = {
    "mnist_fcn": [20, 40, 60, 80, 100],
    "fcn": [20, 40, 60, 80, 100],
}
DATASET_INTERVALS = {}
DEFAULT_INTERVAL = "std"
OUTLIER_FILTER_DATASETS = {"pubmed_rct20k"}
OUTLIER_MODIFIED_Z_THRESHOLD = 1.5

EXCLUDED_PLOT_POINTS = {
    ("pubmed_rct20k", "lowest_k_gradients", 0.05),
    ("pubmed_rct20k", "lowest_k_gradients", 0.1),
    ("pubmed_rct20k", "reverse_caps", 0.05),
    ("pubmed_rct20k", "reverse_caps", 0.1),
    ("newsgroup", "reverse_caps", 0.05),
    ("mnist_fcn", "lowest_k_outputs", 0.05),
    ("fcn", "lowest_k_outputs", 0.05),
}


def load_seed_rows(dataset_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(dataset_dir.glob("seed_*.json")):
        if "_trials_" in path.name:
            continue
        with path.open() as f:
            payload = json.load(f)
        for row in payload.get("results", []):
            if "selector" not in row or "param_ratio" not in row:
                continue
            enriched = dict(row)
            enriched["dataset"] = dataset_dir.name
            rows.append(enriched)
    return rows


def load_retrained_baselines(results_dir: Path, metric: str) -> dict[str, float]:
    baselines = {}
    if not results_dir.exists():
        return baselines

    for dataset_dir in sorted(results_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        values = []
        for path in sorted(dataset_dir.glob("seed_*.json")):
            if "_trials_" in path.name:
                continue
            with path.open() as f:
                payload = json.load(f)
            metrics = payload.get("retrained_baseline", {}).get("metrics", {})
            value = metrics.get(metric)
            if value is not None:
                values.append(float(value))
        if values:
            baselines[dataset_dir.name] = float(np.mean(values))
    return baselines


def get_retrained_baseline(
    retrained_baselines: dict[str, float], dataset: str
) -> float | None:
    if dataset in retrained_baselines:
        return retrained_baselines[dataset]
    for alias in DATASET_ALIASES.get(dataset, []):
        if alias in retrained_baselines:
            return retrained_baselines[alias]
    return None


def dataset_caption(dataset: str, panel_label: str | None = None) -> str:
    caption = DATASET_CAPTIONS.get(dataset, dataset.replace("_", " "))
    if panel_label is None:
        return caption
    return f"({panel_label}) {caption}"


def filter_outliers(values: list[float]) -> tuple[np.ndarray, int]:
    arr = np.asarray(values, dtype=float)
    if len(arr) < 4:
        return arr, 0

    median = np.median(arr)
    mad = np.median(np.abs(arr - median))
    if mad == 0:
        return arr, 0

    modified_z = 0.6745 * (arr - median) / mad
    kept = np.abs(modified_z) <= OUTLIER_MODIFIED_Z_THRESHOLD
    return arr[kept], int((~kept).sum())


def aggregate(rows: list[dict], metric: str) -> dict[tuple[str, float], dict]:
    values: dict[tuple[str, float], list[float]] = defaultdict(list)
    reached_values: dict[tuple[str, float], list[float]] = defaultdict(list)
    missed_values: dict[tuple[str, float], list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if value is None:
            continue
        key = (row["selector"], float(row["param_ratio"]))
        values[key].append(float(value))
        if row.get("reached_target"):
            reached_values[key].append(float(value))
        else:
            missed_values[key].append(float(value))

    summary = {}
    for key, vals in values.items():
        arr = np.asarray(vals, dtype=float)
        reached_arr = np.asarray(reached_values.get(key, []), dtype=float)
        spread_arr = reached_arr
        outlier_n = 0
        selector, _ = key
        group_datasets = {
            row.get("dataset")
            for row in rows
            if row.get("selector") == selector
            and float(row.get("param_ratio")) == key[1]
        }
        if group_datasets & OUTLIER_FILTER_DATASETS:
            spread_arr, outlier_n = filter_outliers(reached_values.get(key, []))
        missed_arr = np.asarray(missed_values.get(key, []), dtype=float)
        summary[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "n": int(len(arr)),
            "reached_n": int(len(reached_arr)),
            "spread_n": int(len(spread_arr)),
            "missed_n": int(len(missed_arr)),
            "outlier_n": outlier_n,
            "reached_mean": float(reached_arr.mean()) if len(reached_arr) > 0 else None,
            "reached_std": float(spread_arr.std(ddof=1))
            if len(spread_arr) > 1
            else 0.0,
            "missed_mean": float(missed_arr.mean()) if len(missed_arr) > 0 else None,
            "missed_std": float(missed_arr.std(ddof=1)) if len(missed_arr) > 1 else 0.0,
        }
    return summary


def ordered_selectors(
    summary: dict[tuple[str, float], dict], ratios: list[float]
) -> list[str]:
    selectors = [s for s in SELECTOR_ORDER if any((s, r) in summary for r in ratios)]
    selectors += sorted({s for s, _ in summary} - set(selectors))
    return selectors


def should_plot_summary_point(
    dataset: str, selector: str, ratio: float, stats: dict
) -> bool:
    key = (dataset, selector, round(ratio, 10))
    return key not in EXCLUDED_PLOT_POINTS and stats["reached_n"] > 0


def compute_reached_spread(stats: dict, interval: str) -> float:
    if interval == "sem":
        return stats["reached_std"] / np.sqrt(stats["spread_n"])
    return stats["reached_std"]


def marker_face_color(color: str) -> tuple[float, float, float]:
    rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
    return tuple(0.02 * rgb + 0.98 * np.ones(3))


def marker_edge_color(color: str) -> tuple[float, float, float]:
    red, green, blue = mcolors.to_rgb(color)
    hue, lightness, saturation = colorsys.rgb_to_hls(red, green, blue)
    saturation = min(1.0, saturation * 1.35 + 0.12)
    lightness = max(0.26, lightness * 0.82)
    return colorsys.hls_to_rgb(hue, lightness, saturation)


def interval_for_dataset(dataset: str) -> str:
    return DATASET_INTERVALS.get(dataset, DEFAULT_INTERVAL)


def pchip_segments(
    xs: np.ndarray,
    ys: np.ndarray,
    spread: np.ndarray,
    enabled: bool,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Build smoothed plot segments without connecting excluded/missing points."""
    segments = []
    finite = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(spread)
    start = None
    for idx, is_finite in enumerate(np.r_[finite, False]):
        if is_finite and start is None:
            start = idx
        elif not is_finite and start is not None:
            x_seg = xs[start:idx]
            y_seg = ys[start:idx]
            spread_seg = spread[start:idx]
            if enabled and len(x_seg) > 1:
                sample_count = (len(x_seg) - 1) * PCHIP_SAMPLES_PER_INTERVAL + 1
                x_smooth = np.linspace(x_seg[0], x_seg[-1], sample_count)
                y_smooth = PchipInterpolator(x_seg, y_seg)(x_smooth)
                spread_smooth = PchipInterpolator(x_seg, spread_seg)(x_smooth)
                segments.append((x_smooth, y_smooth, np.maximum(spread_smooth, 0.0)))
            else:
                segments.append((x_seg, y_seg, np.maximum(spread_seg, 0.0)))
            start = None
    return segments


def draw_background_grid(ax) -> None:
    """Draw grid as normal low-zorder artists to avoid vector backend ordering bugs."""
    ax.grid(False)
    ax.patch.set_zorder(-10)
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    for x in ax.get_xticks():
        if x_min <= x <= x_max:
            ax.axvline(x, label="_nolegend_", clip_on=True, zorder=0, **GRID_KW)
    for y in ax.get_yticks():
        if y_min <= y <= y_max:
            ax.axhline(y, label="_nolegend_", clip_on=True, zorder=0, **GRID_KW)


def draw_dataset_axis(
    ax,
    dataset: str,
    rows: list[dict],
    metric: str,
    smooth_uncertainty: bool,
    retrained_baseline: float | None = None,
    include_legend_labels: bool = False,
    zoom_xlim: tuple[float, float] | None = None,
    zoom_ylim: tuple[float, float] | None = None,
    zoom_box_ylim: tuple[float, float] | None = None,
    zoom_yticks: list[float] | tuple[float, ...] | None = None,
) -> list:
    summary = aggregate(rows, metric)
    interval = interval_for_dataset(dataset)
    ratios = sorted({ratio for _, ratio in summary})
    selectors = ordered_selectors(summary, ratios)
    handles = []
    zoom_data = []

    for selector in selectors:
        selector_ratios = [r for r in ratios if (selector, r) in summary]
        if not any(
            should_plot_summary_point(dataset, selector, r, summary[(selector, r)])
            for r in selector_ratios
        ):
            continue

        xs = selector_ratios
        ys = [
            summary[(selector, r)]["reached_mean"]
            if should_plot_summary_point(dataset, selector, r, summary[(selector, r)])
            else np.nan
            for r in xs
        ]
        spread = [
            compute_reached_spread(summary[(selector, r)], interval)
            if should_plot_summary_point(dataset, selector, r, summary[(selector, r)])
            else np.nan
            for r in xs
        ]
        xs_arr = np.asarray(xs, dtype=float)
        ys_arr = np.asarray(ys, dtype=float)
        spread_arr = np.asarray(spread, dtype=float)
        finite_points = (
            np.isfinite(xs_arr) & np.isfinite(ys_arr) & np.isfinite(spread_arr)
        )
        smooth_segments = pchip_segments(
            xs_arr,
            ys_arr,
            spread_arr,
            enabled=smooth_uncertainty,
        )
        color = COLORS.get(selector)
        marker = MARKERS.get(selector, "o")
        marker_size = MARKER_SIZES.get(selector, 6.6)
        inset_marker_size = INSET_MARKER_SIZES.get(selector, 5.3)
        line_zorder = SELECTOR_LINE_ZORDER.get(selector, 50)
        band_zorder = SELECTOR_BAND_ZORDER.get(selector, 5)
        label = (
            SELECTOR_LABELS.get(selector, selector) if include_legend_labels else None
        )
        handle = Line2D(
            [0],
            [0],
            marker=marker,
            linewidth=2.0,
            markersize=marker_size,
            color=color,
            markerfacecolor=marker_face_color(color),
            markeredgecolor=marker_edge_color(color),
            markeredgewidth=1.15,
            label=label,
        )

        for x_smooth, y_smooth, spread_smooth in smooth_segments:
            ax.plot(
                x_smooth,
                y_smooth,
                linewidth=2.0,
                color=color,
                zorder=line_zorder,
                label="_nolegend_",
            )
            ax.fill_between(
                x_smooth,
                y_smooth - spread_smooth,
                y_smooth + spread_smooth,
                color=color,
                alpha=0.2,
                linewidth=0,
                zorder=band_zorder,
            )
        ax.plot(
            xs_arr[finite_points],
            ys_arr[finite_points],
            marker=marker,
            linestyle="None",
            markersize=marker_size,
            color=color,
            markerfacecolor=marker_face_color(color),
            markeredgecolor=marker_edge_color(color),
            markeredgewidth=1.15,
            zorder=line_zorder + 1,
            label="_nolegend_",
        )
        handles.append(handle)
        zoom_data.append(
            (
                smooth_segments,
                xs_arr[finite_points],
                ys_arr[finite_points],
                color,
                marker,
                line_zorder,
                band_zorder,
                inset_marker_size,
            )
        )
    if retrained_baseline is not None:
        ax.axhline(retrained_baseline, label="_nolegend_", **RETRAINED_LINE_KW)

    ax.set_xticks(MAIN_XTICKS)
    ax.set_xticklabels(MAIN_XTICK_LABELS, fontsize=8)
    ax.set_xlim(*MAIN_XLIM)
    if dataset in DATASET_YLIMS:
        ax.set_ylim(*DATASET_YLIMS[dataset])
    if dataset in DATASET_YTICKS:
        ax.set_yticks(DATASET_YTICKS[dataset])
    ax.tick_params(axis="y", labelsize=8)
    draw_background_grid(ax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if zoom_xlim is not None:
        zoom_xmin, zoom_xmax = zoom_xlim
        zoom_ax = inset_axes(
            ax,
            width="44%",
            height="44%",
            loc="lower right",
            bbox_to_anchor=(-0.055, 0.075, 1.0, 1.0),
            bbox_transform=ax.transAxes,
            borderpad=1.0,
        )
        zoom_y_values = []
        if retrained_baseline is not None:
            zoom_ax.axhline(retrained_baseline, label="_nolegend_", **RETRAINED_LINE_KW)
        for (
            smooth_segments,
            marker_xs,
            marker_ys,
            color,
            marker,
            line_zorder,
            band_zorder,
            inset_marker_size,
        ) in zoom_data:
            for x_smooth, y_smooth, spread_smooth in smooth_segments:
                mask = (zoom_xmin <= x_smooth) & (x_smooth <= zoom_xmax)
                if not np.any(mask):
                    continue
                zoom_ax.plot(
                    x_smooth[mask],
                    y_smooth[mask],
                    linewidth=1.5,
                    color=color,
                    zorder=line_zorder,
                )
                zoom_ax.fill_between(
                    x_smooth[mask],
                    y_smooth[mask] - spread_smooth[mask],
                    y_smooth[mask] + spread_smooth[mask],
                    color=color,
                    alpha=0.13,
                    linewidth=0,
                    zorder=band_zorder,
                )
                zoom_y_values.extend((y_smooth[mask] - spread_smooth[mask]).tolist())
                zoom_y_values.extend((y_smooth[mask] + spread_smooth[mask]).tolist())
            marker_mask = (zoom_xmin <= marker_xs) & (marker_xs <= zoom_xmax)
            if np.any(marker_mask):
                zoom_ax.plot(
                    marker_xs[marker_mask],
                    marker_ys[marker_mask],
                    marker=marker,
                    linestyle="None",
                    markersize=inset_marker_size,
                    color=color,
                    markerfacecolor=marker_face_color(color),
                    markeredgecolor=marker_edge_color(color),
                    markeredgewidth=0.9,
                    zorder=line_zorder + 1,
                )

        zoom_ax.set_xlim(zoom_xmin, zoom_xmax)
        box_ymin = None
        box_ymax = None
        if zoom_ylim is not None:
            zoom_ymin, zoom_ymax = zoom_ylim
            show_fraction_ticks = bool(
                zoom_y_values and max(zoom_y_values) > 1.0 and zoom_ymax <= 1.0
            )
            if show_fraction_ticks:
                zoom_ymin *= 100.0
                zoom_ymax *= 100.0
            box_ymin = zoom_ymin
            box_ymax = zoom_ymax
            zoom_ax.set_ylim(zoom_ymin, zoom_ymax)
            if zoom_yticks is not None:
                tick_positions = [
                    tick * 100.0 if show_fraction_ticks else tick
                    for tick in zoom_yticks
                ]
                zoom_ax.set_yticks(tick_positions)
                zoom_ax.set_yticklabels([f"{tick:.3f}" for tick in zoom_yticks])
            elif show_fraction_ticks:
                zoom_ax.set_yticks(
                    [zoom_ymin, (zoom_ymin + zoom_ymax) / 2.0, zoom_ymax]
                )
                zoom_ax.set_yticklabels(
                    [f"{tick / 100.0:.2f}" for tick in zoom_ax.get_yticks()]
                )
            if zoom_box_ylim is not None:
                box_ymin, box_ymax = zoom_box_ylim
                if show_fraction_ticks:
                    box_ymin *= 100.0
                    box_ymax *= 100.0
            ax.add_patch(
                Rectangle(
                    (zoom_xmin, box_ymin),
                    zoom_xmax - zoom_xmin,
                    box_ymax - box_ymin,
                    fill=False,
                    edgecolor="0.25",
                    linewidth=1.0,
                    linestyle="--",
                    zorder=10,
                )
            )
        elif zoom_y_values:
            ymin = min(zoom_y_values)
            ymax = max(zoom_y_values)
            pad = max((ymax - ymin) * 0.12, 0.1)
            box_ymin = ymin - pad
            box_ymax = ymax + pad
            zoom_ax.set_ylim(ymin - pad, ymax + pad)
            ax.add_patch(
                Rectangle(
                    (zoom_xmin, ymin - pad),
                    zoom_xmax - zoom_xmin,
                    (ymax - ymin) + 2 * pad,
                    fill=False,
                    edgecolor="0.25",
                    linewidth=1.0,
                    linestyle="--",
                    zorder=10,
                )
            )
        zoom_ax.set_xticks([0.05, 0.1, 0.2])
        zoom_ax.set_xticklabels(["5", "10", "20"])
        zoom_ax.tick_params(labelsize=7, pad=1)
        draw_background_grid(zoom_ax)
        if box_ymax is not None:
            for parent_xy, inset_xy in [
                ((zoom_xmin, box_ymin), (0.0, 1.0)),
                ((zoom_xmax, box_ymin), (1.0, 1.0)),
            ]:
                ax.figure.add_artist(
                    ConnectionPatch(
                        xyA=parent_xy,
                        coordsA=ax.transData,
                        axesA=ax,
                        xyB=inset_xy,
                        coordsB=zoom_ax.transAxes,
                        axesB=zoom_ax,
                        color="0.45",
                        linewidth=0.8,
                        clip_on=False,
                        zorder=11,
                    )
                )
    return handles


def plot_combined_grid(
    all_rows: dict[str, list[dict]],
    out_dir: Path,
    metric: str,
    smooth_uncertainty: bool,
    retrained_baselines: dict[str, float],
) -> Path:
    # Fixed layout: [fcn, cifar10, LEGEND]
    #               [svhn, newsgroups, pubmed]
    layout_datasets = [
        ("mnist_fcn", "fcn"),
        ("cifar10_resnet18",),
        None,
        ("svhn_vgg11",),
        ("newsgroup",),
        ("pubmed_rct20k",),
    ]

    def _find_dataset(slot_names: tuple[str, ...] | None) -> str | None:
        if slot_names is None:
            return None
        for dataset_name in slot_names:
            if dataset_name in all_rows:
                return dataset_name
        for dataset_name in slot_names:
            aliases = DATASET_ALIASES.get(dataset_name, [dataset_name])
            for alias in aliases:
                if alias in all_rows:
                    return alias
        return None

    resolved_layout = [_find_dataset(slot) for slot in layout_datasets]
    if not any(dataset is not None for dataset in resolved_layout):
        raise SystemExit("No expected datasets found for combined plotting")

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 6.3), gridspec_kw={"hspace": 0.4})

    flat_axes = axes.ravel()
    legend_handles = None
    legend_labels = None
    panel_labels = iter("abcde")

    for slot_idx, dataset in enumerate(resolved_layout):
        ax = flat_axes[slot_idx]
        if dataset is None:
            ax.axis("off")
            continue

        handles = draw_dataset_axis(
            ax,
            dataset,
            all_rows[dataset],
            metric,
            smooth_uncertainty=smooth_uncertainty,
            retrained_baseline=get_retrained_baseline(retrained_baselines, dataset),
            include_legend_labels=True,
            zoom_xlim=(0.04, 0.21) if dataset == "cifar10_resnet18" else None,
            zoom_ylim=(0.852, 0.860) if dataset == "cifar10_resnet18" else None,
            zoom_box_ylim=(0.84, 0.88) if dataset == "cifar10_resnet18" else None,
            zoom_yticks=(0.852, 0.856, 0.860)
            if dataset == "cifar10_resnet18"
            else None,
        )
        ax.set_xlabel("Parameter ratio (%)", fontsize=9)
        ax.set_ylabel("Retain (%)", fontsize=9)
        ax.text(
            0.5,
            -0.23,
            dataset_caption(dataset, next(panel_labels)),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=12,
            clip_on=False,
        )
        if legend_handles is None:
            legend_handles = handles
            legend_labels = [handle.get_label() for handle in handles]

    for ax in flat_axes[6:]:
        ax.axis("off")

    legend_ax = flat_axes[2]
    legend_ax.axis("off")
    if legend_handles and legend_labels:
        label_to_handle = dict(zip(legend_labels, legend_handles))
        ordered_handles: list = []
        ordered_labels: list[str] = []
        retrained_inserted = False
        for selector in LEGEND_ORDER:
            label = SELECTOR_LABELS.get(selector, selector)
            handle = label_to_handle.get(label)
            if handle is None:
                continue
            ordered_handles.append(handle)
            ordered_labels.append(label)
            if selector == "highest_k_outputs":
                ordered_handles.append(
                    Line2D(
                        [0],
                        [0],
                        label=RETRAINED_LEGEND_LABEL,
                        **RETRAINED_LINE_KW,
                    )
                )
                ordered_labels.append(RETRAINED_LEGEND_LABEL)
                retrained_inserted = True

        if not retrained_inserted:
            ordered_handles.append(
                Line2D([0], [0], label=RETRAINED_LEGEND_LABEL, **RETRAINED_LINE_KW)
            )
            ordered_labels.append(RETRAINED_LEGEND_LABEL)

        ordered_handles = ordered_handles[:]
        ordered_labels = ordered_labels[:]
        legend_ax.legend(
            ordered_handles,
            ordered_labels,
            loc="center",
            frameon=False,
            fontsize=11,
            ncol=2,
            handlelength=2.4,
            labelspacing=1.0,
        )

    out_path = out_dir / f"selection_schemes_{metric}.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--retrained-results-dir",
        type=Path,
        default=Path("../unlearn/results"),
        help="Directory containing JSON files with retrained_baseline metrics.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/plots"))
    parser.add_argument("--metric", default="retain_acc")
    parser.add_argument(
        "--include-small-runs",
        action="store_true",
        help="Include result folders with fewer than two parameter ratios.",
    )
    parser.add_argument(
        "--no-smooth-uncertainty",
        action="store_true",
        help="Draw raw mean and std bands without PCHIP smoothing.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = {}
    for dataset_dir in sorted(args.results_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        rows = load_seed_rows(dataset_dir)
        ratios = {row.get("param_ratio") for row in rows}
        if not args.include_small_runs and len(ratios) < 2:
            continue
        if rows:
            all_rows[dataset_dir.name] = rows

    if not all_rows:
        raise SystemExit(f"No plottable result rows found under {args.results_dir}")

    retrained_baselines = load_retrained_baselines(
        args.retrained_results_dir, args.metric
    )

    out_path = plot_combined_grid(
        all_rows,
        args.out_dir,
        args.metric,
        smooth_uncertainty=not args.no_smooth_uncertainty,
        retrained_baselines=retrained_baselines,
    )
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
