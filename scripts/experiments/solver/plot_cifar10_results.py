#!/usr/bin/env python3
"""Plot CIFAR-10 solver residual, memory, and elapsed time results."""

from __future__ import annotations

import argparse
import colorsys
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.interpolate import PchipInterpolator  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS = (
    PROJECT_ROOT
    / "scripts"
    / "experiments"
    / "solver"
    / "results"
    / "cifar10_solver_accuracy_scaling.csv"
)
DEFAULT_OUT = (
    PROJECT_ROOT
    / "scripts"
    / "experiments"
    / "solver"
    / "results"
    / "cifar10_solver_accuracy_scaling_metrics.pdf"
)

SIZE_ORDER = ["25k", "100k", "500k", "2m", "5m", "10m", "20m"]
SOLVER_ORDER = [
    "p_lissa_full",
    "p_lissa_0p1",
    "lissa",
    "cg",
    "schulz",
    "lanczos",
    "kfac",
    "ekfac",
]
SOLVER_LABELS = {
    "p_lissa_full": "p-LiSSA full",
    "p_lissa_0p1": "p-LiSSA 0.1",
    "lissa": "LiSSA",
    "cg": "CG",
    "schulz": "Schulz",
    "lanczos": "Lanczos",
    "kfac": "KFAC",
    "ekfac": "EKFAC",
}
COLORS = {
    "p_lissa_full": "#073b8e",
    "p_lissa_0p1": "#2575f4",
    "lissa": "#5cbcbf",
    "cg": "#666666",
    "schulz": "#8b5cf6",
    "lanczos": "#f4aa62",
    "kfac": "#e8563d",
    "ekfac": "#b21424",
}
MARKERS = {
    "p_lissa_full": "o",
    "p_lissa_0p1": "8",
    "lissa": "h",
    "cg": "s",
    "schulz": "P",
    "lanczos": "v",
    "kfac": "D",
    "ekfac": "^",
}
GRID_KW = {"linestyle": (0, (5, 5)), "linewidth": 0.5, "color": "#e0e0e0"}
PCHIP_SAMPLES_PER_INTERVAL = 24
METRICS = [
    ("residual_pct", "Residual (%)", True),
    ("memory_mb", "Memory (MB)", True),
    ("elapsed_s", "Elapsed time (s)", False),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read CIFAR-10 solver results and plot residual (%), memory (MB), "
            "and elapsed time (s)."
        )
    )
    parser.add_argument(
        "results",
        nargs="?",
        type=Path,
        default=DEFAULT_RESULTS,
        help="CSV or JSON results file written by scripts/experiments/solver/cifar10.py.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--png",
        action="store_true",
        help="Also write a PNG next to the primary output file.",
    )
    return parser.parse_args()


def as_float(value: Any) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return list(payload.get("results", []))

    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def marker_face_color(color: str) -> tuple[float, float, float]:
    rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
    return tuple(0.02 * rgb + 0.98 * np.ones(3))


def marker_edge_color(color: str) -> tuple[float, float, float]:
    red, green, blue = mcolors.to_rgb(color)
    hue, lightness, saturation = colorsys.rgb_to_hls(red, green, blue)
    saturation = min(1.0, saturation * 1.35 + 0.12)
    lightness = max(0.26, lightness * 0.82)
    return colorsys.hls_to_rgb(hue, lightness, saturation)


def draw_background_grid(ax) -> None:
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


def row_memory_mb(row: dict[str, Any]) -> float:
    cuda_peak_mb = as_float(row.get("cuda_peak_mb"))
    if math.isfinite(cuda_peak_mb):
        return cuda_peak_mb
    return as_float(row.get("python_peak_mb"))


def enrich_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched = []
    for row in rows:
        item = dict(row)
        item["residual_pct"] = as_float(row.get("eval_residual")) * 100.0
        item["memory_mb"] = row_memory_mb(row)
        item["elapsed_s"] = as_float(row.get("time_sec"))
        item["actual_params"] = as_float(row.get("actual_params"))
        enriched.append(item)
    return enriched


def ordered_sizes(rows: list[dict[str, Any]]) -> list[str]:
    present = {str(row.get("size", "")) for row in rows}
    sizes = [size for size in SIZE_ORDER if size in present]
    sizes += sorted(present - set(sizes))
    return sizes


def ordered_solvers(rows: list[dict[str, Any]]) -> list[str]:
    present = {str(row.get("solver", "")) for row in rows}
    solvers = [solver for solver in SOLVER_ORDER if solver in present]
    solvers += sorted(present - set(solvers))
    return solvers


def build_series(
    rows: list[dict[str, Any]], sizes: list[str], solver: str, metric: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_size: dict[str, list[float]] = {size: [] for size in sizes}
    for row in rows:
        if row.get("solver") != solver:
            continue
        if row.get("status") != "ok":
            continue
        size = str(row.get("size", ""))
        if size not in by_size:
            continue
        value = as_float(row.get(metric))
        if math.isfinite(value):
            by_size[size].append(value)

    xs = np.arange(len(sizes), dtype=float)
    ys = []
    spread = []
    for size in sizes:
        values = np.asarray(by_size[size], dtype=float)
        ys.append(float(values.mean()) if len(values) else math.nan)
        spread.append(float(values.std(ddof=1)) if len(values) > 1 else 0.0)
    return xs, np.asarray(ys, dtype=float), np.asarray(spread, dtype=float)


def smooth_segments(
    xs: np.ndarray,
    ys: np.ndarray,
    spread: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
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
            if len(x_seg) > 1:
                sample_count = (len(x_seg) - 1) * PCHIP_SAMPLES_PER_INTERVAL + 1
                x_smooth = np.linspace(x_seg[0], x_seg[-1], sample_count)
                y_smooth = PchipInterpolator(x_seg, y_seg)(x_smooth)
                spread_smooth = PchipInterpolator(x_seg, spread_seg)(x_smooth)
                segments.append(
                    (x_smooth, y_smooth, np.maximum(spread_smooth, 0.0))
                )
            else:
                segments.append((x_seg, y_seg, np.maximum(spread_seg, 0.0)))
            start = None
    return segments


def plot_results(rows: list[dict[str, Any]], out_path: Path, write_png: bool) -> Path:
    rows = enrich_rows(rows)
    sizes = ordered_sizes(rows)
    solvers = ordered_solvers(rows)
    if not sizes or not solvers:
        raise SystemExit("No plottable CIFAR-10 solver rows found.")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12.0, 7.2),
        gridspec_kw={"hspace": 0.45, "wspace": 0.28},
    )
    metric_axes = [axes[0, 0], axes[0, 1], axes[1, 0]]
    legend_ax = axes[1, 1]
    legend_ax.axis("off")
    handles = []

    for ax, (metric, ylabel, use_log_y) in zip(metric_axes, METRICS):
        for solver_index, solver in enumerate(solvers):
            if metric == "elapsed_s" and solver == "schulz":
                continue
            color = COLORS.get(solver, "#444444")
            marker = MARKERS.get(solver, "o")
            xs, ys, spread = build_series(rows, sizes, solver, metric)
            finite = np.isfinite(ys)
            if not np.any(finite):
                continue
            line_zorder = 50 + len(solvers) - solver_index
            band_zorder = 10 - solver_index * 0.1
            for x_smooth, y_smooth, spread_smooth in smooth_segments(xs, ys, spread):
                ax.plot(
                    x_smooth,
                    y_smooth,
                    linewidth=2.0,
                    color=color,
                    zorder=line_zorder,
                    label="_nolegend_",
                )
                if np.any(spread_smooth > 0.0):
                    ax.fill_between(
                        x_smooth,
                        y_smooth - spread_smooth,
                        y_smooth + spread_smooth,
                        color=color,
                        alpha=0.16,
                        linewidth=0,
                        zorder=band_zorder,
                    )
            ax.plot(
                xs[finite],
                ys[finite],
                linestyle="None",
                marker=marker,
                markersize=7.2,
                color=color,
                markerfacecolor=marker_face_color(color),
                markeredgecolor=marker_edge_color(color),
                markeredgewidth=1.1,
                zorder=line_zorder + 1,
                label="_nolegend_",
            )
            if np.any(spread[finite] > 0.0):
                ax.fill_between(
                    xs[finite],
                    ys[finite] - spread[finite],
                    ys[finite] + spread[finite],
                    color=color,
                    alpha=0.08,
                    linewidth=0,
                    zorder=band_zorder,
                )
            if metric == METRICS[0][0]:
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker=marker,
                        linewidth=2.0,
                        markersize=7.2,
                        color=color,
                        markerfacecolor=marker_face_color(color),
                        markeredgecolor=marker_edge_color(color),
                        markeredgewidth=1.1,
                        label=SOLVER_LABELS.get(solver, solver),
                    )
                )

        ax.set_xticks(np.arange(len(sizes)))
        ax.set_xticklabels(sizes, fontsize=8)
        ax.set_xlabel("Model size", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        if use_log_y:
            ax.set_yscale("log")
        if metric == "memory_mb":
            ax.set_ylim(1e2, 4e4)
        if metric == "elapsed_s":
            ax.set_yscale("log")
        ax.tick_params(axis="y", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.margins(x=0.04, y=0.12)
        draw_background_grid(ax)

    legend_ax.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="center",
        frameon=False,
        fontsize=10,
        ncol=2,
        handlelength=2.4,
        labelspacing=1.0,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if write_png:
        png_path = out_path.with_suffix(".png")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    out_path = plot_results(load_rows(args.results), args.out, args.png)
    print(f"Wrote: {out_path}")
    if args.png:
        print(f"Wrote: {out_path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
