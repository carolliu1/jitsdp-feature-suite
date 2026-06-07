from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


KAMEI_FEATURES = {
    "fix",
    "ns",
    "nd",
    "nf",
    "entropy",
    "la",
    "ld",
    "lt",
    "ndev",
    "age",
    "nuc",
    "exp",
    "rexp",
    "sexp",
}
KAMEI_COLOR = "#4C78A8"
NEW_COLOR = "#F28E2B"
COMBINED_COLOR = "#59A14F"


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def is_kamei(name: object) -> bool:
    return str(name) in KAMEI_FEATURES


def family_label(name: object) -> str:
    return "kamei" if is_kamei(name) else "new"


def family_colors(names: pd.Series | list[str]) -> list[str]:
    return [KAMEI_COLOR if is_kamei(name) else NEW_COLOR for name in names]


def add_family_legend(ax: plt.Axes) -> None:
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=KAMEI_COLOR, label="Kamei features"),
        plt.Rectangle((0, 0), 1, 1, color=NEW_COLOR, label="New features/concepts"),
    ]
    ax.legend(handles=handles, loc="best", frameon=False)


def stage3_family_color(name: object) -> str:
    if str(name) == "kamei":
        return KAMEI_COLOR
    if str(name) == "new":
        return NEW_COLOR
    return COMBINED_COLOR


def stage3_metric_label(metric: str) -> str:
    labels = {
        "f1": "F1",
        "mcc": "MCC",
        "gmean": "G-mean",
        "recall0": "Recall 0",
        "recall1": "Recall 1",
    }
    return labels.get(metric, metric)


def plot_stage1_consensus(consensus: pd.DataFrame, output_dir: Path, top_n: int | None = None) -> None:
    if consensus.empty or "average_rank" not in consensus.columns:
        return
    data = consensus.copy()
    if "consensus_rank" not in data.columns:
        data["consensus_rank"] = data["average_rank"].rank(method="dense", ascending=True).astype(int)
    data = data.sort_values(["consensus_rank", "average_rank", "concept"])
    if top_n is not None:
        data = data.head(top_n)
    data = data.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.32 * len(data))))
    ax.barh(data["concept"], data["average_rank"], color=family_colors(data["concept"]))
    ax.set_xlabel("Average NPSK rank (lower is better)")
    ax.set_ylabel("Feature concept")
    ax.set_title("Stage 1 Consensus Ranking")
    ax.grid(axis="x", alpha=0.25)
    max_rank = data["average_rank"].max()
    if pd.notna(max_rank):
        ax.set_xlim(0, float(max_rank) + max(3.0, float(max_rank) * 0.18))
    for y, (_, row) in enumerate(data.iterrows()):
        value = row["average_rank"]
        if pd.notna(value):
            ax.text(
                float(value) + 0.05,
                y,
                f"Rank {int(row['consensus_rank'])} ({value:.2f})",
                va="center",
                fontsize=8,
            )
    add_family_legend(ax)
    save_figure(fig, output_dir, "stage1_consensus_all_features")


def plot_stage1_rank_heatmap(consensus: pd.DataFrame, output_dir: Path, top_n: int | None = None) -> None:
    method_columns = [
        column
        for column in consensus.columns
        if column not in {"concept", "average_rank", "rank_std", "consensus_rank", "feature_family"}
    ]
    if consensus.empty or not method_columns:
        return
    data = consensus.sort_values(["average_rank", "concept"])
    if top_n is not None:
        data = data.head(top_n)
    matrix = data[method_columns].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(1.1 * len(method_columns) + 4, max(4, 0.32 * len(data))))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis_r")
    ax.set_xticks(np.arange(len(method_columns)), labels=method_columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(data)), labels=data["concept"])
    for label, concept in zip(ax.get_yticklabels(), data["concept"]):
        label.set_color(KAMEI_COLOR if is_kamei(concept) else NEW_COLOR)
    ax.set_title("Stage 1 NPSK Ranks by Univariate Method")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Rank (lower is better)")

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            if not np.isnan(value):
                ax.text(col, row, f"{value:.0f}", ha="center", va="center", color="white", fontsize=8)

    save_figure(fig, output_dir, "stage1_rank_heatmap")


def plot_stage2_top_rank_frequency(top_rank_frequency: pd.DataFrame, output_dir: Path, top_n: int | None = None) -> None:
    if top_rank_frequency.empty or "rank1_frequency" not in top_rank_frequency.columns:
        return
    data = top_rank_frequency.sort_values(["rank1_frequency", "mean_rank"], ascending=[False, True])
    if top_n is not None:
        data = data.head(top_n)
    data = data.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.32 * len(data))))
    ax.barh(data["concept"], data["rank1_frequency"], color=family_colors(data["concept"]))
    ax.set_xlabel("Rank-1 frequency across datasets")
    ax.set_ylabel("Feature concept")
    ax.set_xlim(0, 1)
    ax.set_title("Stage 2 Top-Rank Frequency")
    ax.grid(axis="x", alpha=0.25)
    for y, (_, row) in enumerate(data.iterrows()):
        frequency = row["rank1_frequency"]
        if pd.isna(frequency):
            continue
        label = f"{frequency:.2f}"
        if "mean_rank" in row and pd.notna(row["mean_rank"]):
            label = f"{label}, mean rank {row['mean_rank']:.2f}"
        ax.text(min(float(frequency) + 0.015, 1.01), y, label, va="center", fontsize=8)
    ax.set_xlim(0, 1.12)
    add_family_legend(ax)
    save_figure(fig, output_dir, "stage2_top_rank_frequency")


def plot_stage2_rank_matrix(rank_matrix: pd.DataFrame, output_dir: Path, top_n: int | None = None) -> None:
    if rank_matrix.empty:
        return
    mean_rank = rank_matrix.mean(axis=0, skipna=True).sort_values()
    selected = mean_rank.index.tolist() if top_n is None else mean_rank.head(top_n).index.tolist()
    data = rank_matrix[selected]
    matrix = data.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(max(8, 0.38 * len(selected)), max(4, 0.34 * len(data))))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis_r")
    ax.set_xticks(np.arange(len(selected)), labels=selected, rotation=60, ha="right")
    for label, concept in zip(ax.get_xticklabels(), selected):
        label.set_color(KAMEI_COLOR if is_kamei(concept) else NEW_COLOR)
    ax.set_yticks(np.arange(len(data)), labels=data.index)
    ax.set_title("Stage 2 Local NPSK Rank Matrix")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Rank (lower is better)")

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            if not np.isnan(value):
                ax.text(col, row, f"{value:.0f}", ha="center", va="center", color="white", fontsize=7)

    save_figure(fig, output_dir, "stage2_rank_matrix")


def plot_stage2_npsk_ranks(
    ranking: pd.DataFrame,
    output_dir: Path,
    stem: str,
    title: str,
    rank_column: str = "rank",
) -> None:
    if ranking.empty or "concept" not in ranking.columns or rank_column not in ranking.columns:
        return
    data = ranking[["concept", rank_column]].dropna().copy()
    if data.empty:
        return
    data = data.sort_values([rank_column, "concept"], ascending=[False, False])
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.28 * len(data))))
    ax.barh(data["concept"], data[rank_column], color=family_colors(data["concept"]))
    ax.set_xlabel("NPSK rank (lower is better)")
    ax.set_ylabel("Feature concept")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    max_rank = data[rank_column].max()
    if pd.notna(max_rank):
        ax.set_xlim(0, float(max_rank) + 0.75)
    for y, value in enumerate(data[rank_column]):
        if pd.notna(value):
            ax.text(float(value) + 0.05, y, f"{value:.0f}", va="center", fontsize=8)
    add_family_legend(ax)
    save_figure(fig, output_dir, stem)


def plot_stage2_npsk_ranks_long(
    ranking_long: pd.DataFrame,
    output_dir: Path,
    stem: str,
    title: str,
) -> None:
    required = {"dataset", "concept", "rank"}
    if ranking_long.empty or not required.issubset(ranking_long.columns):
        return
    matrix = ranking_long.pivot_table(
        index="dataset",
        columns="concept",
        values="rank",
        aggfunc="first",
    )
    if matrix.empty:
        return
    ordered_columns = matrix.mean(axis=0, skipna=True).sort_values().index.tolist()
    preferred_rows = [row for row in matrix.index if row != "ALL_DATASETS"]
    if "ALL_DATASETS" in matrix.index:
        preferred_rows.append("ALL_DATASETS")
    matrix = matrix.loc[preferred_rows, ordered_columns]

    values = matrix.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(max(10, 0.38 * len(ordered_columns)), max(4.5, 0.35 * len(matrix))))
    image = ax.imshow(values, aspect="auto", cmap="viridis_r")
    ax.set_xticks(np.arange(len(ordered_columns)), labels=ordered_columns, rotation=60, ha="right")
    for label, concept in zip(ax.get_xticklabels(), ordered_columns):
        label.set_color(KAMEI_COLOR if is_kamei(concept) else NEW_COLOR)
    ax.set_yticks(np.arange(len(matrix.index)), labels=matrix.index)
    ax.set_title(title)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("NPSK rank (lower is better)")
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if not np.isnan(value):
                ax.text(col, row, f"{value:.0f}", ha="center", va="center", color="white", fontsize=7)
    save_figure(fig, output_dir, stem)


def plot_stage3_family_performance(summary: pd.DataFrame, output_dir: Path) -> None:
    if summary.empty:
        return
    for metric in ("mcc", "gmean", "recall0", "recall1", "f1"):
        metric_column = f"{metric}_mean"
        if metric_column not in summary.columns:
            continue
        _plot_stage3_metric_bar(summary, output_dir, metric, metric_column)
        _plot_stage3_metric_heatmap(summary, output_dir, metric, metric_column)


def plot_stage3_topk_performance(summary: pd.DataFrame, output_dir: Path) -> None:
    if summary.empty:
        return
    metrics = ("gmean", "recall0", "recall1", "mcc", "f1")
    available = [metric for metric in metrics if f"{metric}_mean" in summary.columns]
    if not available:
        return

    _plot_stage3_topk_all_metrics(summary, output_dir, available)
    for metric in available:
        metric_column = f"{metric}_mean"
        _plot_stage3_topk_metric_line(summary, output_dir, metric, metric_column)
        _plot_stage3_topk_median_across_datasets(summary, output_dir, metric, metric_column)
        _plot_stage3_topk_metric_heatmap(summary, output_dir, metric, metric_column)


def _ordered_experiments(experiments: pd.Series) -> list[str]:
    preferred = ["kamei", "new", "combined"]
    existing = [experiment for experiment in preferred if experiment in set(experiments)]
    existing.extend(sorted(set(experiments) - set(existing)))
    return existing


def _ordered_stage3_experiments(experiments: pd.Series) -> list[str]:
    values = [str(experiment) for experiment in experiments.dropna().unique()]
    ordered = []
    if "kamei_14" in values:
        ordered.append("kamei_14")
    top_values = []
    for value in values:
        if value.startswith("top_"):
            try:
                top_values.append((int(value.split("_", 1)[1]), value))
            except ValueError:
                continue
    ordered.extend(value for _, value in sorted(top_values))
    ordered.extend(sorted(set(values) - set(ordered)))
    return ordered


def _plot_stage3_topk_all_metrics(summary: pd.DataFrame, output_dir: Path, metrics: tuple[str, ...] | list[str]) -> None:
    ordered = _ordered_stage3_experiments(summary["experiment"])
    aggregate = summary.groupby("experiment", as_index=False)[[f"{metric}_mean" for metric in metrics]].median()
    aggregate["experiment"] = pd.Categorical(aggregate["experiment"], categories=ordered, ordered=True)
    aggregate = aggregate.sort_values("experiment")
    x_positions = np.arange(len(aggregate))
    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(aggregate)), 4.8))
    for metric in metrics:
        metric_column = f"{metric}_mean"
        ax.plot(
            x_positions,
            aggregate[metric_column],
            marker="o",
            linewidth=1.8,
            label=stage3_metric_label(metric),
        )
        for x, y in zip(x_positions, aggregate[metric_column]):
            if pd.notna(y):
                ax.text(x, y, f"{y:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xlabel("Feature set")
    ax.set_ylabel("Median performance across datasets")
    ax.set_title("Stage 3 Baseline and Top-k Median Performance Across Metrics")
    ax.set_xticks(x_positions, labels=aggregate["experiment"], rotation=30, ha="right")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, output_dir, "stage3_topk_all_metrics_summary")


def _plot_stage3_topk_metric_line(summary: pd.DataFrame, output_dir: Path, metric: str, metric_column: str) -> None:
    ordered = _ordered_stage3_experiments(summary["experiment"])
    aggregate = summary.groupby("experiment", as_index=False)[metric_column].median()
    aggregate["experiment"] = pd.Categorical(aggregate["experiment"], categories=ordered, ordered=True)
    aggregate = aggregate.sort_values("experiment")
    x_positions = np.arange(len(aggregate))
    fig, ax = plt.subplots(figsize=(max(7.5, 0.7 * len(aggregate)), 4.5))
    ax.plot(x_positions, aggregate[metric_column], marker="o", linewidth=1.8)
    for x, y in zip(x_positions, aggregate[metric_column]):
        if pd.notna(y):
            ax.text(x, y, f"{y:.3f}", ha="center", va="bottom", fontsize=8)
    metric_label = stage3_metric_label(metric)
    ax.set_xlabel("Feature set")
    ax.set_ylabel(f"Median {metric_label} across datasets")
    ax.set_title(f"Stage 3 Baseline and Top-k Median Performance: {metric_label}")
    ax.set_xticks(x_positions, labels=aggregate["experiment"], rotation=30, ha="right")
    ax.grid(alpha=0.25)
    save_figure(fig, output_dir, f"stage3_topk_{metric}_summary")


def _plot_stage3_topk_median_across_datasets(
    summary: pd.DataFrame,
    output_dir: Path,
    metric: str,
    metric_column: str,
) -> None:
    data = summary[summary["experiment"].astype(str).str.startswith("top_")].copy()
    if data.empty:
        return
    data["topk_percent"] = pd.to_numeric(
        data["experiment"].astype(str).str.replace("top_", "", regex=False),
        errors="coerce",
    )
    data = data.dropna(subset=["topk_percent", metric_column])
    if data.empty:
        return

    aggregate = (
        data.groupby("topk_percent", as_index=False)[metric_column]
        .median()
        .sort_values("topk_percent")
    )
    if aggregate.empty:
        return

    x_values = aggregate["topk_percent"].astype(int)
    y_values = aggregate[metric_column]
    metric_label = stage3_metric_label(metric)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(x_values, y_values, marker="o", linewidth=1.8)
    for x, y in zip(x_values, y_values):
        if pd.notna(y):
            ax.text(x, y, f"{y:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Top-k NPSK rank groups (%)")
    ax.set_ylabel(f"Median {metric_label} across datasets")
    ax.set_title(f"Stage 3 Top-k Median Across Datasets: {metric_label}")
    ax.set_xticks(x_values)
    ax.grid(alpha=0.25)
    save_figure(fig, output_dir, f"stage3_topk_{metric}_median_across_datasets")


def _plot_stage3_topk_metric_heatmap(summary: pd.DataFrame, output_dir: Path, metric: str, metric_column: str) -> None:
    matrix = summary.pivot_table(
        index="dataset",
        columns="experiment",
        values=metric_column,
        aggfunc="mean",
    )
    if matrix.empty:
        return
    matrix = matrix.reindex(columns=_ordered_stage3_experiments(pd.Series(matrix.columns)))
    values = matrix.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(max(8, 0.72 * len(matrix.columns)), max(4, 0.34 * len(matrix.index))))
    image = ax.imshow(values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(matrix.columns)), labels=matrix.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)), labels=matrix.index)
    metric_label = stage3_metric_label(metric)
    ax.set_title(f"Stage 3 Baseline and Top-k Performance by Dataset: {metric_label}")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(metric_label)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if not np.isnan(value):
                ax.text(col, row, f"{value:.3f}", ha="center", va="center", color="white", fontsize=7)
    save_figure(fig, output_dir, f"stage3_topk_{metric}_by_dataset")


STAGE4_EXPERIMENT_ORDER = (
    "kamei_14",
    "kamei_method_cc",
    "kamei_method_structure",
    "kamei_halstead",
    "kamei_file_metrics",
    "kamei_coupling",
    "kamei_global_var",
    "kamei_commit_stats",
)

STAGE4_EXPERIMENT_LABELS = {
    "kamei_14": "Kamei 14",
    "kamei_method_cc": "Kamei + method_cc",
    "kamei_method_structure": "Kamei + method structure",
    "kamei_halstead": "Kamei + Halstead",
    "kamei_file_metrics": "Kamei + file metrics",
    "kamei_coupling": "Kamei + coupling",
    "kamei_global_var": "Kamei + global var",
    "kamei_commit_stats": "Kamei + commit stats",
}


def plot_stage4_category_performance(summary: pd.DataFrame, output_dir: Path) -> None:
    if summary.empty:
        return
    metrics = ("gmean", "recall0", "recall1", "mcc", "f1")
    available = [metric for metric in metrics if f"{metric}_mean" in summary.columns]
    if not available:
        return

    _plot_stage4_all_metrics(summary, output_dir, available)
    for metric in available:
        metric_column = f"{metric}_mean"
        _plot_stage4_metric_bar(summary, output_dir, metric, metric_column)
        _plot_stage4_metric_heatmap(summary, output_dir, metric, metric_column)
        _plot_stage4_metric_median(summary, output_dir, metric, metric_column)


def _ordered_stage4_experiments(experiments: pd.Series) -> list[str]:
    values = [str(experiment) for experiment in experiments.dropna().unique()]
    ordered = [experiment for experiment in STAGE4_EXPERIMENT_ORDER if experiment in values]
    ordered.extend(sorted(set(values) - set(ordered)))
    return ordered


def _stage4_labels(experiments: pd.Series | list[str]) -> list[str]:
    return [STAGE4_EXPERIMENT_LABELS.get(str(experiment), str(experiment)) for experiment in experiments]


def _plot_stage4_all_metrics(summary: pd.DataFrame, output_dir: Path, metrics: tuple[str, ...] | list[str]) -> None:
    ordered = _ordered_stage4_experiments(summary["experiment"])
    aggregate = summary.groupby("experiment", as_index=False)[[f"{metric}_mean" for metric in metrics]].median()
    aggregate["experiment"] = pd.Categorical(aggregate["experiment"], categories=ordered, ordered=True)
    aggregate = aggregate.sort_values("experiment")
    x_positions = np.arange(len(aggregate))
    fig, ax = plt.subplots(figsize=(max(9, 0.9 * len(aggregate)), 4.8))
    for metric in metrics:
        metric_column = f"{metric}_mean"
        ax.plot(
            x_positions,
            aggregate[metric_column],
            marker="o",
            linewidth=1.8,
            label=stage3_metric_label(metric),
        )
        for x, y in zip(x_positions, aggregate[metric_column]):
            if pd.notna(y):
                ax.text(x, y, f"{y:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xlabel("Feature set")
    ax.set_ylabel("Median performance across datasets")
    ax.set_title("Stage 4 Feature Category Median Comparison Across Metrics")
    ax.set_xticks(x_positions, labels=_stage4_labels(aggregate["experiment"]), rotation=35, ha="right")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, output_dir, "stage4_all_metrics_summary")


def _plot_stage4_metric_bar(summary: pd.DataFrame, output_dir: Path, metric: str, metric_column: str) -> None:
    ordered = _ordered_stage4_experiments(summary["experiment"])
    aggregate = summary.groupby("experiment", as_index=False)[metric_column].median()
    aggregate["experiment"] = pd.Categorical(aggregate["experiment"], categories=ordered, ordered=True)
    aggregate = aggregate.sort_values("experiment")

    fig, ax = plt.subplots(figsize=(max(8.5, 0.9 * len(aggregate)), 4.7))
    bars = ax.bar(np.arange(len(aggregate)), aggregate[metric_column], color=NEW_COLOR)
    metric_label = stage3_metric_label(metric)
    ax.set_xlabel("Feature set")
    ax.set_ylabel(f"Median {metric_label} across datasets")
    ax.set_title(f"Stage 4 Feature Category Median Comparison: {metric_label}")
    ax.set_xticks(np.arange(len(aggregate)), labels=_stage4_labels(aggregate["experiment"]), rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    max_value = aggregate[metric_column].max()
    if pd.notna(max_value):
        ax.set_ylim(0, float(max_value) * 1.18 if max_value > 0 else 1)
    for bar, value in zip(bars, aggregate[metric_column]):
        if pd.notna(value):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    save_figure(fig, output_dir, f"stage4_{metric}_summary")


def _plot_stage4_metric_heatmap(summary: pd.DataFrame, output_dir: Path, metric: str, metric_column: str) -> None:
    matrix = summary.pivot_table(
        index="dataset",
        columns="experiment",
        values=metric_column,
        aggfunc="mean",
    )
    if matrix.empty:
        return
    matrix = matrix.reindex(columns=_ordered_stage4_experiments(pd.Series(matrix.columns)))
    values = matrix.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(max(9, 0.95 * len(matrix.columns)), max(4, 0.34 * len(matrix.index))))
    image = ax.imshow(values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(matrix.columns)), labels=_stage4_labels(matrix.columns), rotation=35, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)), labels=matrix.index)
    metric_label = stage3_metric_label(metric)
    ax.set_title(f"Stage 4 Feature Category Comparison by Dataset: {metric_label}")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(metric_label)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if not np.isnan(value):
                ax.text(col, row, f"{value:.3f}", ha="center", va="center", color="white", fontsize=7)
    save_figure(fig, output_dir, f"stage4_{metric}_by_dataset")


def _plot_stage4_metric_median(summary: pd.DataFrame, output_dir: Path, metric: str, metric_column: str) -> None:
    ordered = _ordered_stage4_experiments(summary["experiment"])
    aggregate = summary.groupby("experiment", as_index=False)[metric_column].median()
    aggregate["experiment"] = pd.Categorical(aggregate["experiment"], categories=ordered, ordered=True)
    aggregate = aggregate.sort_values("experiment")
    x_positions = np.arange(len(aggregate))
    fig, ax = plt.subplots(figsize=(max(8.5, 0.9 * len(aggregate)), 4.7))
    ax.plot(x_positions, aggregate[metric_column], marker="o", linewidth=1.8)
    for x, y in zip(x_positions, aggregate[metric_column]):
        if pd.notna(y):
            ax.text(x, y, f"{y:.3f}", ha="center", va="bottom", fontsize=8)
    metric_label = stage3_metric_label(metric)
    ax.set_xlabel("Feature set")
    ax.set_ylabel(f"Median {metric_label} across datasets")
    ax.set_title(f"Stage 4 Median Across Datasets: {metric_label}")
    ax.set_xticks(x_positions, labels=_stage4_labels(aggregate["experiment"]), rotation=35, ha="right")
    ax.grid(alpha=0.25)
    save_figure(fig, output_dir, f"stage4_{metric}_median_across_datasets")


def _plot_stage3_metric_bar(summary: pd.DataFrame, output_dir: Path, metric: str, metric_column: str) -> None:
    data = summary.groupby("experiment", as_index=False)[metric_column].mean()
    ordered = _ordered_experiments(data["experiment"])
    data["experiment"] = pd.Categorical(data["experiment"], categories=ordered, ordered=True)
    data = data.sort_values("experiment")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(
        data["experiment"].astype(str),
        data[metric_column],
        color=[stage3_family_color(experiment) for experiment in data["experiment"].astype(str)],
    )
    ax.set_xlabel("Feature family")
    metric_label = stage3_metric_label(metric)
    ax.set_ylabel(metric_label)
    ax.set_title(f"Stage 3 Family Performance: {metric_label}")
    ax.grid(axis="y", alpha=0.25)
    max_value = data[metric_column].max()
    if pd.notna(max_value):
        ax.set_ylim(0, float(max_value) * 1.15 if max_value > 0 else 1)
    for bar, value in zip(bars, data[metric_column]):
        if pd.notna(value):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    save_figure(fig, output_dir, f"stage3_family_{metric}_summary")


def _plot_stage3_metric_heatmap(summary: pd.DataFrame, output_dir: Path, metric: str, metric_column: str) -> None:
    ordered_experiments = _ordered_experiments(summary["experiment"])
    matrix = summary.pivot_table(
        index="dataset",
        columns="experiment",
        values=metric_column,
        aggfunc="mean",
    )
    matrix = matrix.reindex(columns=ordered_experiments)
    if matrix.empty:
        return

    values = matrix.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(max(6, 1.8 * len(matrix.columns)), max(4, 0.35 * len(matrix.index))))
    image = ax.imshow(values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(matrix.columns)), labels=matrix.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)), labels=matrix.index)
    metric_label = stage3_metric_label(metric)
    ax.set_title(f"Stage 3 Family Performance by Dataset: {metric_label}")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(metric_label)

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if not np.isnan(value):
                ax.text(col, row, f"{value:.3f}", ha="center", va="center", color="white", fontsize=8)
    save_figure(fig, output_dir, f"stage3_family_{metric}_by_dataset")
