#!/usr/bin/env python3
"""Build the model-edit comparison LaTeX table from saved JSON results."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATASETS = [
    ("mnist_fcn", "MNIST", "FCN"),
    ("cifar10_resnet18", "CIFAR-10", "ResNet-18"),
    ("svhn_vgg11", "SVHN", "VGG-11"),
    ("newsgroup", "Newsgroups", "BERT-L2-H128"),
    ("pubmed_rct20k", "PubMed 20k RCT", "BERT-L2-H128"),
]

METHODS = [
    ("retrain", "Retrain"),
    ("classical_if", "Classic IF"),
    ("second_order_if", "Second-order IF"),
    ("tracin", "TracIn"),
    ("hypeinf", "HyperInf"),
    ("datainf", "DataInf"),
    ("freezing", "Schioppa"),
    ("ekfac", "EKFAC"),
    ("gif", "Ours"),
]


@dataclass(frozen=True)
class MetricSummary:
    retain_mean: float | None
    retain_std: float | None
    unlearn_mean: float | None
    unlearn_std: float | None
    count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the LaTeX model-edit comparison table from results/*.json."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Directory containing per-dataset result subdirectories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the generated LaTeX table.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=2,
        help="Number of decimal places in table cells.",
    )
    parser.add_argument(
        "--std",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include standard deviation as mean $\\pm$ std.",
    )
    parser.add_argument(
        "--missing",
        default="--",
        help="Cell text to use when a method/dataset result is unavailable.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any expected dataset directory or method result is missing.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def summarize_dataset(dataset_key: str, dataset_dir: Path) -> dict[str, MetricSummary]:
    rows: dict[str, dict[str, list[float]]] = {
        method: {"retain": [], "unlearn": []} for method, _ in METHODS
    }

    for result_path in sorted(dataset_dir.glob("seed_*.json")):
        payload = load_json(result_path)

        retrained_metrics = (
            payload.get("retrained_baseline", {}).get("metrics", {})
            if isinstance(payload.get("retrained_baseline"), dict)
            else {}
        )
        append_metrics(rows["retrain"], retrained_metrics)

        for result in payload.get("results", []):
            if not isinstance(result, dict):
                continue
            if result.get("status") != "ok":
                continue
            method = result.get("method")
            if method not in rows:
                continue
            if should_exclude_result(dataset_key, result):
                continue
            append_metrics(rows[method], result)

    return {
        method: MetricSummary(
            retain_mean=mean_std(metrics["retain"])[0],
            retain_std=mean_std(metrics["retain"])[1],
            unlearn_mean=mean_std(metrics["unlearn"])[0],
            unlearn_std=mean_std(metrics["unlearn"])[1],
            count=min(len(metrics["retain"]), len(metrics["unlearn"])),
        )
        for method, metrics in rows.items()
    }


def should_exclude_result(dataset_key: str, result: dict[str, Any]) -> bool:
    """Drop known failed unlearning outliers before aggregation."""
    if dataset_key != "svhn_vgg11":
        return False
    if result.get("method") != "freezing":
        return False
    self_acc = finite_float(result.get("self_acc"))
    return self_acc is not None and self_acc >= 90.0


def append_metrics(target: dict[str, list[float]], metrics: dict[str, Any]) -> None:
    retain_acc = finite_float(metrics.get("retain_acc"))
    self_acc = finite_float(metrics.get("self_acc"))
    if retain_acc is not None:
        target["retain"].append(retain_acc)
    if self_acc is not None:
        target["unlearn"].append(self_acc)


def format_value(
    mean: float | None,
    std: float | None,
    *,
    precision: int,
    include_std: bool,
    missing: str,
) -> str:
    if mean is None:
        return missing
    value = f"{mean:.{precision}f}"
    if include_std and std is not None:
        value = (
            rf"{value} {{\scriptsize \textcolor{{gray}}"
            rf"{{$\pm$ {std:.{precision}f}}}}}"
        )
    return value


def maybe_bold(value: str, is_best: bool) -> str:
    if not is_best:
        return value
    return rf"\textbf{{{value}}}"


def almost_equal(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return False
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def best_methods_for_dataset(
    dataset_summary: dict[str, MetricSummary],
) -> tuple[str | None, str | None]:
    candidate_methods = [
        method
        for method, _ in METHODS
        if method != "retrain" and method in dataset_summary
    ]
    best_retain = choose_best_method(
        candidate_methods,
        dataset_summary,
        metric="retain",
        higher_is_better=True,
    )
    best_unlearn = choose_best_method(
        candidate_methods,
        dataset_summary,
        metric="unlearn",
        higher_is_better=False,
    )
    return best_retain, best_unlearn


def choose_best_method(
    methods: list[str],
    dataset_summary: dict[str, MetricSummary],
    *,
    metric: str,
    higher_is_better: bool,
) -> str | None:
    values: list[tuple[str, float]] = []
    for method in methods:
        summary = dataset_summary[method]
        value = summary.retain_mean if metric == "retain" else summary.unlearn_mean
        if value is not None:
            values.append((method, value))
    if not values:
        return None

    best_value = (
        max(value for _, value in values)
        if higher_is_better
        else min(value for _, value in values)
    )
    tied_methods = [
        method for method, value in values if almost_equal(value, best_value)
    ]
    if "gif" in tied_methods:
        return "gif"
    return tied_methods[0]


def format_pair(
    summary: MetricSummary,
    args: argparse.Namespace,
    *,
    is_best_retain: bool,
    is_best_unlearn: bool,
) -> tuple[str, str]:
    retain = format_value(
        summary.retain_mean,
        summary.retain_std,
        precision=args.precision,
        include_std=args.std,
        missing=args.missing,
    )
    unlearn = format_value(
        summary.unlearn_mean,
        summary.unlearn_std,
        precision=args.precision,
        include_std=args.std,
        missing=args.missing,
    )
    retain = maybe_bold(retain, is_best_retain)
    unlearn = maybe_bold(unlearn, is_best_unlearn)
    return retain, unlearn


def validate_summaries(
    summaries: dict[str, dict[str, MetricSummary]], args: argparse.Namespace
) -> None:
    missing: list[str] = []
    for dataset_key, dataset_label, _ in DATASETS:
        if dataset_key not in summaries:
            missing.append(f"{dataset_label}: missing dataset directory")
            continue
        for method_key, method_label in METHODS:
            if summaries[dataset_key][method_key].count == 0:
                missing.append(f"{dataset_label}/{method_label}: missing result")
    if missing and args.strict:
        joined = "\n  - ".join(missing)
        raise SystemExit(f"Missing expected results:\n  - {joined}")


def build_table(
    summaries: dict[str, dict[str, MetricSummary]], args: argparse.Namespace
) -> str:
    dataset_headers = [
        rf"& \multicolumn{{2}}{{c}}{{\shortstack{{{dataset_label}\\{{\scriptsize {model_label}}}}}}}"
        for _, dataset_label, model_label in DATASETS
    ]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Comprehensive evaluation of model editing ability for each influence-function-based scheme.",
        r"For every dataset, we report two metrics: \textit{Retain} ($\uparrow$) and \textit{Unlearn} ($\downarrow$).}",
        r"\label{tab:model_edit_comparison}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{adjustbox}{width=\textwidth}",
        r"\begin{tabular}{l cc cc cc cc cc}",
        r"\toprule",
        r"\multirow{2}{*}{Method}",
        *dataset_headers[:-1],
        dataset_headers[-1] + r" \\",
        r"\cmidrule(lr){2-3}",
        r"\cmidrule(lr){4-5}",
        r"\cmidrule(lr){6-7}",
        r"\cmidrule(lr){8-9}",
        r"\cmidrule(lr){10-11}",
        r"& Retain ($\uparrow$) & Unlearn ($\downarrow$)",
        r"& Retain ($\uparrow$) & Unlearn ($\downarrow$)",
        r"& Retain ($\uparrow$) & Unlearn ($\downarrow$)",
        r"& Retain ($\uparrow$) & Unlearn ($\downarrow$)",
        r"& Retain ($\uparrow$) & Unlearn ($\downarrow$) \\",
        r"\midrule",
        "",
        r"\arrayrulecolor{lightgray}",
    ]

    for index, (method_key, method_label) in enumerate(METHODS):
        cells: list[str] = []
        for dataset_key, _, _ in DATASETS:
            dataset_summary = summaries.get(dataset_key, {})
            summary = dataset_summary.get(
                method_key, MetricSummary(None, None, None, None, 0)
            )
            best_retain_method, best_unlearn_method = best_methods_for_dataset(
                dataset_summary
            )
            retain, unlearn = format_pair(
                summary,
                args,
                is_best_retain=method_key == best_retain_method,
                is_best_unlearn=method_key == best_unlearn_method,
            )
            cells.extend([retain, unlearn])

        lines.append(format_method_row(method_label, cells))
        if index == 0:
            lines.append(r"\arrayrulecolor{black}")
            lines.append(r"\cmidrule(l{2pt}r{2pt}){1-11}")
            lines.append(r"\arrayrulecolor{lightgray}")
            lines.append("")
        elif index == len(METHODS) - 2:
            lines.append(r"\arrayrulecolor{black}")
            lines.append(r"\cmidrule[0.75pt](l{2pt}r{2pt}){1-11}")
            lines.append(r"\arrayrulecolor{lightgray}")
        elif index < len(METHODS) - 1:
            lines.append(r"\cmidrule(l{2pt}r{2pt}){1-1}\cmidrule(l{2pt}r{2pt}){2-11}")
            lines.append("")

    lines.extend(
        [
            "",
            r"\arrayrulecolor{black}",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{adjustbox}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines)


def format_method_row(method_label: str, cells: list[str]) -> str:
    return f"{method_label:<16} & " + " & ".join(cells) + r" \\"


def main() -> None:
    args = parse_args()
    summaries: dict[str, dict[str, MetricSummary]] = {}
    for dataset_key, _, _ in DATASETS:
        dataset_dir = args.results_dir / dataset_key
        if dataset_dir.is_dir():
            summaries[dataset_key] = summarize_dataset(dataset_key, dataset_dir)

    validate_summaries(summaries, args)
    table = build_table(summaries, args)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
