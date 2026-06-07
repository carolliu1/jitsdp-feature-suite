from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import Stage3Config
from .data import PreparedDataset
from .modeling import RandomForestService, TimeForwardSplitter
from .plots import plot_stage4_category_performance
from .project_presets import KAMEI_FEATURE_COLUMNS
from .stage3 import STAGE3_METRIC_COLUMNS


STAT_SUFFIXES = ("mean", "max", "min", "std")
STAGE4_METRIC_COLUMNS = STAGE3_METRIC_COLUMNS


@dataclass(frozen=True)
class Stage4FeatureSet:
    experiment: str
    category: str
    added_original_metrics: tuple[str, ...]
    feature_columns: tuple[str, ...]

    @property
    def selected_feature_count(self) -> int:
        return len(self.feature_columns)


def _stat_columns(metric: str) -> tuple[str, ...]:
    return tuple(f"current_{metric}_{suffix}" for suffix in STAT_SUFFIXES)


def _with_kamei(*columns: str) -> tuple[str, ...]:
    return (*KAMEI_FEATURE_COLUMNS, *columns)


def stage4_feature_sets() -> list[Stage4FeatureSet]:
    halstead_metrics = (
        "method_halstead_n1",
        "method_halstead_n2",
        "method_halstead_N1",
        "method_halstead_N2",
        "method_halstead_length",
        "method_halstead_vocabulary",
        "method_halstead_volume",
        "method_halstead_difficulty",
        "method_halstead_effort",
        "method_halstead_bugs",
        "method_halstead_time",
    )
    return [
        Stage4FeatureSet(
            experiment="kamei_14",
            category="Kamei change features",
            added_original_metrics=(),
            feature_columns=KAMEI_FEATURE_COLUMNS,
        ),
        Stage4FeatureSet(
            experiment="kamei_method_cc",
            category="Method-level complexity",
            added_original_metrics=("method_cc",),
            feature_columns=_with_kamei(*_stat_columns("method_cc")),
        ),
        Stage4FeatureSet(
            experiment="kamei_method_structure",
            category="Method-level structure",
            added_original_metrics=("method_nargs", "method_nexits"),
            feature_columns=_with_kamei(
                *_stat_columns("method_nargs"),
                *_stat_columns("method_nexits"),
            ),
        ),
        Stage4FeatureSet(
            experiment="kamei_halstead",
            category="Method-level Halstead",
            added_original_metrics=halstead_metrics,
            feature_columns=_with_kamei(
                *(column for metric in halstead_metrics for column in _stat_columns(metric)),
            ),
        ),
        Stage4FeatureSet(
            experiment="kamei_file_metrics",
            category="File-level metrics",
            added_original_metrics=("file_cloc", "file_mi", "file_nom"),
            feature_columns=_with_kamei(
                *_stat_columns("file_cloc"),
                *_stat_columns("file_mi"),
                *_stat_columns("file_nom"),
            ),
        ),
        Stage4FeatureSet(
            experiment="kamei_coupling",
            category="Method-level coupling",
            added_original_metrics=("method_in_coupling", "method_out_coupling"),
            feature_columns=_with_kamei(
                *_stat_columns("method_in_coupling"),
                *_stat_columns("method_out_coupling"),
            ),
        ),
        Stage4FeatureSet(
            experiment="kamei_global_var",
            category="Method-level external state",
            added_original_metrics=("method_global_var_count",),
            feature_columns=_with_kamei(*_stat_columns("method_global_var_count")),
        ),
        Stage4FeatureSet(
            experiment="kamei_commit_stats",
            category="Commit-level statistics",
            added_original_metrics=(
                "modified_method_count",
                "new_file_count",
                "new_file_ratio",
            ),
            feature_columns=_with_kamei(
                "modified_method_count_current",
                "current_new_file_count",
                "current_new_file_ratio",
            ),
        ),
    ]


class Stage4CategoryComparisonAnalyzer:
    """Prediction-performance comparison for Kamei plus each new feature category.

    The modeling protocol is intentionally the same as Stage 3: time-forward
    windows, random-forest hyperparameter tuning inside each training window,
    class-balanced random forests, and repeated training with different seeds.
    """

    def __init__(self, config: Stage3Config | None = None) -> None:
        self.config = config or Stage3Config()
        self.splitter = TimeForwardSplitter(self.config.time_windows)
        self.rf_service = RandomForestService(self.config.random_forest)

    def run(
        self,
        datasets: list[PreparedDataset],
        output_dir: Path | None = None,
    ) -> pd.DataFrame:
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)

        feature_sets = stage4_feature_sets()
        selected_features = self._selected_features_table(feature_sets)

        rows: list[dict[str, object]] = []
        window_rows: list[dict[str, object]] = []
        total_datasets = len(datasets)
        for feature_set in feature_sets:
            print(
                f"[stage4] {feature_set.experiment}: evaluating "
                f"{feature_set.selected_feature_count} fixed features.",
                flush=True,
            )
            for dataset_index, dataset in enumerate(datasets, start=1):
                missing = sorted(set(feature_set.feature_columns) - set(dataset.feature_columns))
                if missing:
                    raise ValueError(
                        f"{dataset.name}: Stage 4 {feature_set.experiment} features are missing: {missing}"
                    )
                print(
                    f"[stage4] Dataset {dataset_index}/{total_datasets}: "
                    f"{dataset.name}, {feature_set.experiment}.",
                    flush=True,
                )
                performance, windows = self._evaluate_feature_set(dataset, feature_set)
                rows.extend(performance)
                window_rows.extend(windows)

        result = pd.DataFrame(rows)
        time_windows = pd.DataFrame(window_rows)
        summary = self._summary(result)

        if output_dir is not None:
            selected_features.to_csv(output_dir / "stage4_selected_features.csv", index=False)
            result.to_csv(output_dir / "stage4_performance.csv", index=False)
            summary.to_csv(output_dir / "stage4_performance_summary.csv", index=False)
            time_windows.to_csv(output_dir / "stage4_time_windows.csv", index=False)
            self._write_dataset_metric_mean_table(summary, output_dir)
            self._write_median_table(summary, output_dir)
            plot_stage4_category_performance(summary, output_dir / "figures")
            print("[stage4] CSV and figures written.", flush=True)

        return result

    def _selected_features_table(self, feature_sets: list[Stage4FeatureSet]) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for feature_set in feature_sets:
            for feature_rank, feature in enumerate(feature_set.feature_columns, start=1):
                rows.append(
                    {
                        "experiment": feature_set.experiment,
                        "category": feature_set.category,
                        "added_original_metrics": ",".join(feature_set.added_original_metrics),
                        "selected_feature_count": feature_set.selected_feature_count,
                        "feature_rank_within_set": feature_rank,
                        "feature": feature,
                    }
                )
        return pd.DataFrame(rows)

    def _evaluate_feature_set(
        self,
        dataset: PreparedDataset,
        feature_set: Stage4FeatureSet,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        rows: list[dict[str, object]] = []
        window_rows: list[dict[str, object]] = []
        feature_columns = list(feature_set.feature_columns)

        for window in self.splitter.split(dataset.frame):
            train = dataset.frame.iloc[window.train_start : window.train_end]
            test = dataset.frame.iloc[window.test_start : window.test_end]
            X_train = train[feature_columns]
            y_train = train[dataset.label_column].astype(int)
            X_test = test[feature_columns]
            y_test = test[dataset.label_column].astype(int)
            window_record = window.to_record(dataset.name, dataset.frame, dataset.date_column)
            window_record["experiment"] = feature_set.experiment
            window_record["category"] = feature_set.category
            window_record["selected_feature_count"] = feature_set.selected_feature_count
            window_record["used_for_modeling"] = True
            window_record["skip_reason"] = ""

            if y_train.nunique() < 2 or y_test.nunique() < 2:
                window_record["used_for_modeling"] = False
                window_record["skip_reason"] = "single_class_train_or_test"
                window_rows.append(window_record)
                print(
                    f"[stage4] {dataset.name}: {feature_set.experiment}, "
                    f"window {window.window_id} skipped ({window_record['skip_reason']}).",
                    flush=True,
                )
                continue

            window_rows.append(window_record)

            experiment_seed = self._experiment_seed(feature_set.experiment)
            seed = self.config.random_state + window.window_id + experiment_seed
            params = self.rf_service.tune(X_train, y_train, random_state=seed)
            print(
                f"[stage4] {dataset.name}: {feature_set.experiment}, "
                f"window {window.window_id} tuned params {params}.",
                flush=True,
            )

            for repeat in range(1, self.config.repeats + 1):
                run_seed = seed + repeat
                model = self.rf_service.fit(X_train, y_train, params=params, random_state=run_seed)
                metrics = self.rf_service.evaluate(model, X_test, y_test)
                selected_metrics = {metric: metrics[metric] for metric in STAGE4_METRIC_COLUMNS}
                rows.append(
                    {
                        **window_record,
                        "repeat": repeat,
                        **selected_metrics,
                    }
                )

        return rows, window_rows

    def _experiment_seed(self, experiment: str) -> int:
        ordered = self._experiment_order()
        return ordered.index(experiment) * 100 if experiment in ordered else 0

    def _experiment_order(self) -> list[str]:
        return [feature_set.experiment for feature_set in stage4_feature_sets()]

    def _summary(self, result: pd.DataFrame) -> pd.DataFrame:
        if result.empty:
            return pd.DataFrame()
        available = [column for column in STAGE4_METRIC_COLUMNS if column in result.columns]
        summary = (
            result.groupby(
                ["dataset", "experiment", "category", "selected_feature_count"],
                as_index=False,
            )[available]
            .agg(["mean", "std"])
            .reset_index()
        )
        summary.columns = [
            "_".join(part for part in column if part)
            if isinstance(column, tuple)
            else str(column)
            for column in summary.columns.to_flat_index()
        ]
        return summary

    def _write_dataset_metric_mean_table(self, summary: pd.DataFrame, output_dir: Path) -> None:
        if summary.empty:
            return
        metric_mean_columns = [f"{metric}_mean" for metric in STAGE4_METRIC_COLUMNS]
        available = [column for column in metric_mean_columns if column in summary.columns]
        table = summary[
            ["dataset", "experiment", "category", "selected_feature_count", *available]
        ].copy()
        table = table.rename(
            columns={f"{metric}_mean": metric for metric in STAGE4_METRIC_COLUMNS}
        )
        table.to_csv(output_dir / "stage4_dataset_metric_means.csv", index=False)

        ordered_columns = [column for column in self._experiment_order() if column in set(table["experiment"])]
        for metric in STAGE4_METRIC_COLUMNS:
            if metric not in table.columns:
                continue
            matrix = table.pivot_table(
                index="dataset",
                columns="experiment",
                values=metric,
                aggfunc="mean",
            )
            matrix = matrix.reindex(columns=ordered_columns)
            matrix.to_csv(output_dir / f"stage4_{metric}_matrix.csv")

    def _write_median_table(self, summary: pd.DataFrame, output_dir: Path) -> None:
        if summary.empty:
            return
        rows: list[dict[str, object]] = []
        for experiment, group in summary.groupby("experiment"):
            row: dict[str, object] = {"experiment": experiment}
            if "category" in group.columns:
                row["category"] = group["category"].iloc[0]
            if "selected_feature_count" in group.columns:
                row["selected_feature_count"] = int(group["selected_feature_count"].iloc[0])
            for metric in STAGE4_METRIC_COLUMNS:
                metric_column = f"{metric}_mean"
                if metric_column in group.columns:
                    row[f"{metric}_median_across_datasets"] = group[metric_column].median()
            rows.append(row)

        if not rows:
            return
        order = self._experiment_order()
        table = pd.DataFrame(rows)
        table["experiment"] = pd.Categorical(table["experiment"], categories=order, ordered=True)
        table = table.sort_values("experiment")
        table.to_csv(output_dir / "stage4_median_across_datasets.csv", index=False)
