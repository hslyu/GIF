#!/usr/bin/env python3
"""Plot CIFAR-10 solver residual, memory, and elapsed time results."""

from __future__ import annotations

import argparse
import colorsys
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib import patheffects as pe  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.interpolate import PchipInterpolator  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PRETENDARD_FONT_DIR = Path("/fast/hslyu/font")
PRETENDARD_REGULAR_PATH = PRETENDARD_FONT_DIR / "Pretendard-Regular.ttf"
PRETENDARD_MEDIUM_PATH = PRETENDARD_FONT_DIR / "Pretendard-Medium.ttf"

TICK_FONTSIZE = 12
AXIS_LABEL_FONTSIZE = 12
LEGEND_FONTSIZE = 10
DAMPING_LABEL_FONTSIZE = 10
PLOT_FIGSIZE = (18.0, 3)
PLOT_WSPACE = 0.35
PLOT_WIDTH_RATIOS = [1.0, 1.0, 1.0, 0.3]

DEFAULT_SEED_RESULTS = (
    PROJECT_ROOT / "scripts" / "experiments" / "solver" / "results" / "cifar10"
)
DEFAULT_OUT = (
    Path("/home/hslyu/research/rework/GIF_latex/Figures") / "solver_cifar10.pdf"
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
    "p_lissa_full": "P-LiSSA",
    "p_lissa_0p1": "P-LiSSA with 10% param",
    "lissa": "LiSSA",
    "cg": "Conjugate gradient",
    "schulz": "Schulz iteration",
    "lanczos": "Lanczos iteration",
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
    ("residual", "Regularized residual", True),
    ("memory_mb", "Memory (MB)", True),
    ("elapsed_s", "Elapsed time (s)", False),
]
DAMPING_ANNOTATIONS = [
    {
        "line_solver": "ekfac",
        "value_solvers": ("ekfac", "kfac"),
        "x_min": 2e4,
        "x_max": 1e5,
    },
    {
        "line_solver": "lissa",
        "value_solvers": ("lissa",),
        "x_min": 1e5,
        "x_max": 5e5,
    },
    {
        "line_solver": "schulz",
        "value_solvers": ("schulz",),
        "x_min": 5e5,
        "x_max": 2e6,
    },
]

for font_path in (PRETENDARD_REGULAR_PATH, PRETENDARD_MEDIUM_PATH):
    if font_path.exists():
        fm.fontManager.addfont(str(font_path))


def pretendard_regular(size: float) -> fm.FontProperties:
    if PRETENDARD_REGULAR_PATH.exists():
        return fm.FontProperties(fname=str(PRETENDARD_REGULAR_PATH), size=size)
    return fm.FontProperties(size=size)


def pretendard_medium(size: float) -> fm.FontProperties:
    if PRETENDARD_MEDIUM_PATH.exists():
        return fm.FontProperties(fname=str(PRETENDARD_MEDIUM_PATH), size=size)
    return fm.FontProperties(size=size, weight="medium")


def apply_tick_font(ax, size: float = TICK_FONTSIZE) -> None:
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontproperties(pretendard_regular(size))


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
        default=DEFAULT_SEED_RESULTS,
        help=(
            "Seed results directory containing per-seed JSON files, or an explicit "
            "CSV/JSON results file written by scripts/experiments/solver/cifar10.py."
        ),
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


def load_result_file(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return list(payload.get("results", []))

    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def result_row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("seed", "")),
        str(row.get("size", "")),
        str(row.get("solver", "")),
    )


def load_seed_rows(path: Path) -> list[dict[str, Any]]:
    files: list[Path] = []
    for seed_dir in sorted(path.glob("seed_*")):
        if not seed_dir.is_dir():
            continue
        files.extend(sorted(seed_dir.glob("*.json")))

    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    for result_file in files:
        for row in load_result_file(result_file):
            merged[result_row_key(row)] = row
    return list(merged.values())


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        return load_seed_rows(path)
    return load_result_file(path)


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
        item["residual"] = as_float(row.get("eval_residual"))
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


def ordered_param_counts(
    rows: list[dict[str, Any]], sizes: list[str]
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        size = str(row.get("size", ""))
        if size not in sizes:
            continue
        value = as_float(row.get("actual_params"))
        if math.isfinite(value):
            grouped[size].append(value)

    params: dict[str, float] = {}
    for size in sizes:
        values = np.asarray(grouped.get(size, []), dtype=float)
        if len(values):
            params[size] = float(np.median(values))
    return params


def ordered_solvers(rows: list[dict[str, Any]]) -> list[str]:
    present = {str(row.get("solver", "")) for row in rows}
    solvers = [solver for solver in SOLVER_ORDER if solver in present]
    solvers += sorted(present - set(solvers))
    return solvers


def format_damping(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    if value == 0.0:
        return "0"
    if 1e-3 <= abs(value) < 1e3:
        return f"{value:g}".replace(".", r"\!.\!")
    mantissa, exponent = f"{value:.0e}".split("e")
    if mantissa == "1":
        return rf"10\!^{{{int(exponent)}}}"
    return rf"{mantissa}\!\cdot\!10\!^{{{int(exponent)}}}"


def solver_damping_label(
    rows: list[dict[str, Any]],
    solvers: tuple[str, ...],
) -> str | None:
    values = sorted(
        {
            value
            for row in rows
            if str(row.get("solver", "")) in solvers
            and math.isfinite(value := as_float(row.get("damping")))
            and value != 0.0
        }
    )
    if not values:
        return None
    return (
        "$\\lambda\\!\\!=\\!\\!"
        + r"\!/".join(format_damping(value) for value in values)
        + "$"
    )


def damping_annotation_point(
    segments: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    x_min: float,
    x_max: float,
    log_y: bool,
) -> tuple[float, float] | None:
    target_x = math.sqrt(x_min * x_max)
    candidates: list[tuple[float, float]] = []
    for x_smooth, y_smooth, _ in segments:
        mask = (x_smooth >= x_min) & (x_smooth <= x_max)
        if not np.any(mask):
            continue
        x_values = x_smooth[mask]
        y_values = y_smooth[mask]
        index = int(np.argmin(np.abs(np.log(x_values) - math.log(target_x))))
        y_value = (
            float(np.power(10.0, y_values[index])) if log_y else float(y_values[index])
        )
        candidates.append((float(x_values[index]), y_value))
    if not candidates:
        return None
    return min(candidates, key=lambda item: abs(math.log(item[0]) - math.log(target_x)))


def add_damping_annotation(
    ax,
    rows: list[dict[str, Any]],
    solver: str,
    segments: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    log_y: bool,
    color: str,
) -> None:
    for spec in DAMPING_ANNOTATIONS:
        if solver != spec["line_solver"]:
            continue
        label = solver_damping_label(
            rows,
            spec["value_solvers"],
        )
        point = damping_annotation_point(
            segments,
            x_min=float(spec["x_min"]),
            x_max=float(spec["x_max"]),
            log_y=log_y,
        )
        if label is None or point is None:
            return
        text = ax.annotate(
            label,
            xy=point,
            xytext=(0, 0),
            textcoords="offset points",
            color=color,
            fontproperties=pretendard_medium(DAMPING_LABEL_FONTSIZE),
            fontsize=DAMPING_LABEL_FONTSIZE,
            ha="center",
            va="center",
            zorder=300,
            clip_on=False,
        )
        text.set_path_effects([pe.withStroke(linewidth=4.2, foreground="white")])
        return


def build_series(
    rows: list[dict[str, Any]],
    sizes: list[str],
    size_params: dict[str, float],
    solver: str,
    metric: str,
    *,
    log_y: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_size: dict[str, list[float]] = {size: [] for size in sizes}
    for row in rows:
        if row.get("solver") != solver:
            continue
        size = str(row.get("size", ""))
        if size not in by_size or size not in size_params:
            continue
        value = as_float(row.get(metric))
        if math.isfinite(value) and (not log_y or value > 0.0):
            if log_y:
                value = math.log10(value)
            by_size[size].append(value)

    xs = np.asarray(
        [size_params[size] for size in sizes if size in size_params], dtype=float
    )
    ys = []
    spread = []
    for size in sizes:
        if size not in size_params:
            continue
        values = np.asarray(by_size[size], dtype=float)
        ys.append(float(values.mean()) if len(values) else math.nan)
        spread.append(float(values.std(ddof=1)) if len(values) > 1 else 0.0)
    return xs, np.asarray(ys, dtype=float), np.asarray(spread, dtype=float)


def elapsed_time_limits(rows: list[dict[str, Any]]) -> tuple[float, float]:
    return 0.0, 350.0


def format_param_count(value: float) -> str:
    if not math.isfinite(value) or value <= 0:
        return ""

    exponent = int(math.floor(math.log10(value)))
    mantissa = value / (10**exponent)
    if math.isclose(mantissa, 1.0, rel_tol=1e-6, abs_tol=1e-9):
        return rf"$10^{{{exponent}}}$"

    rounded = int(round(mantissa))
    if rounded >= 10:
        return rf"$10^{{{exponent + 1}}}$"
    return rf"${rounded}\!\cdot\!10^{{{exponent}}}$"


def smooth_segments(
    xs: np.ndarray,
    ys: np.ndarray,
    spread: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    segments = []
    finite = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(spread) & (xs > 0.0)
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
                x_seg_log = np.log10(x_seg)
                x_smooth_log = np.linspace(x_seg_log[0], x_seg_log[-1], sample_count)
                x_smooth = np.power(10.0, x_smooth_log)
                y_smooth = PchipInterpolator(x_seg_log, y_seg)(x_smooth_log)
                spread_smooth = PchipInterpolator(x_seg_log, spread_seg)(x_smooth_log)
                segments.append((x_smooth, y_smooth, np.maximum(spread_smooth, 0.0)))
            else:
                segments.append((x_seg, y_seg, np.maximum(spread_seg, 0.0)))
            start = None
    return segments


def plot_results(rows: list[dict[str, Any]], out_path: Path, write_png: bool) -> Path:
    rows = enrich_rows(rows)
    sizes = ordered_sizes(rows)
    size_params = ordered_param_counts(rows, sizes)
    sizes = [size for size in sizes if size in size_params]
    solvers = ordered_solvers(rows)
    if not sizes or not solvers:
        raise SystemExit("No plottable CIFAR-10 solver rows found.")
    elapsed_ymin, elapsed_ymax = elapsed_time_limits(rows)

    fig, axes = plt.subplots(
        1,
        4,
        figsize=PLOT_FIGSIZE,
        gridspec_kw={"wspace": PLOT_WSPACE, "width_ratios": PLOT_WIDTH_RATIOS},
    )
    metric_axes = axes[:3]
    legend_ax = axes[3]
    legend_ax.axis("off")
    handles = []

    for ax, (metric, ylabel, use_log_y) in zip(metric_axes, METRICS):
        for solver_index, solver in enumerate(solvers):
            if metric == "elapsed_s" and solver == "schulz":
                continue
            color = COLORS.get(solver, "#444444")
            marker = MARKERS.get(solver, "o")
            xs, ys, spread = build_series(
                rows, sizes, size_params, solver, metric, log_y=use_log_y
            )
            finite = np.isfinite(ys)
            if not np.any(finite):
                continue
            line_zorder = 50 + len(solvers) - solver_index
            segments = smooth_segments(xs, ys, spread)
            for x_smooth, y_smooth, spread_smooth in segments:
                if use_log_y:
                    y_plot = np.power(10.0, y_smooth)
                    lower = np.power(10.0, y_smooth - spread_smooth)
                    upper = np.power(10.0, y_smooth + spread_smooth)
                else:
                    y_plot = y_smooth
                    lower = y_smooth - spread_smooth
                    upper = y_smooth + spread_smooth
                ax.plot(
                    x_smooth,
                    y_plot,
                    linewidth=2.0,
                    color=color,
                    zorder=line_zorder,
                    label="_nolegend_",
                )
                if np.any(spread_smooth > 0.0):
                    ax.fill_between(
                        x_smooth,
                        lower,
                        upper,
                        color=color,
                        alpha=0.16,
                        linewidth=0,
                        zorder=10 - solver_index * 0.1,
                    )
            ax.plot(
                xs[finite],
                np.power(10.0, ys[finite]) if use_log_y else ys[finite],
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
            if metric == "residual":
                add_damping_annotation(
                    ax,
                    rows,
                    solver,
                    segments,
                    log_y=use_log_y,
                    color=color,
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

        ax.set_xscale("log")
        xs = np.asarray([size_params[size] for size in sizes], dtype=float)
        ax.set_xticks(xs)
        ax.set_xticklabels([format_param_count(size_params[size]) for size in sizes])
        apply_tick_font(ax)
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_rotation_mode("anchor")
            label.set_horizontalalignment("right")
        ax.set_xlabel(
            "Parameter count", fontproperties=pretendard_medium(AXIS_LABEL_FONTSIZE)
        )
        ax.set_ylabel(ylabel, fontproperties=pretendard_medium(AXIS_LABEL_FONTSIZE))
        if use_log_y:
            ax.set_yscale("log")
        if metric == "memory_mb":
            ax.set_ylim(1e2, 1e5)
        if metric == "elapsed_s":
            ax.set_ylim(elapsed_ymin, elapsed_ymax)
            ax.set_yticks([0, 100, 200, 300])
        ax.set_xlim(xs.min() * 0.85, xs.max() * 1.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.margins(x=0.04, y=0.12)
        draw_background_grid(ax)

    legend_ax.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="center",
        bbox_to_anchor=(0.5, 0.5),
        frameon=True,
        fancybox=True,
        facecolor="none",
        edgecolor="#383838",
        framealpha=1.0,
        prop=pretendard_medium(LEGEND_FONTSIZE),
        ncol=1,
        handlelength=2.4,
        labelspacing=1.0,
        borderpad=0.6,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
    )
    if write_png:
        png_path = out_path.with_suffix(".png")
        fig.savefig(
            png_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
            transparent=True,
        )
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
