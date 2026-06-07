from __future__ import annotations

from math import ceil
from pathlib import Path

import pandas as pd

from .config import Stage3Config
from .data import PreparedDataset
from .modeling import RandomForestService, TimeForwardSplitter
from .plots import family_label, plot_stage3_topk_performance
from .project_presets import KAMEI_FEATURE_COLUMNS


STAGE3_METRIC_COLUMNS = ("gmean", "recall0", "recall1", "mcc", "f1")
KAMEI_BASELINE_EXPERIMENT = "kamei_14"


class Stage3PerformanceAnalyzer:
    """Prediction-performance experiments using fixed top-k rank-group subsets.

    Stage 3 uses the final Stage 2-B raw-feature MDI/NPSK ranking. For every
    top-k percentage, it selects the top-k percent of NPSK rank groups. All
    features in a selected rank group are kept, so statistically tied features
    are not split across different feature subsets.
    """

    def __init__(self, config: Stage3Config | None = None) -> None:
        self.config = config or Stage3Config()
        self.splitter = TimeForwardSplitter(self.config.time_windows)
        self.rf_service = RandomForestService(self.config.random_forest)

    def run_topk_experiment(
        self,
        datasets: list[PreparedDataset],
        feature_ranking: pd.DataFrame,
        output_dir: Path | None = None,
    ) -> pd.DataFrame:
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)

        ranking = self._final_feature_ranking(feature_ranking, datasets)
        selected_features = self._selected_feature_sets(ranking)

        rows: list[dict[str, object]] = []
        window_rows: list[dict[str, object]] = []
        total_datasets = len(datasets)
        for experiment in self._experiment_order():
            features = selected_features[selected_features["experiment"] == experiment]
            if features.empty:
                continue
            feature_columns = features["feature"].tolist()
            print(
                f"[stage3] {experiment}: evaluating {len(feature_columns)} fixed features.",
                flush=True,
            )
            for dataset_index, dataset in enumerate(datasets, start=1):
                missing = sorted(set(feature_columns) - set(dataset.feature_columns))
                if missing:
                    raise ValueError(
                        f"{dataset.name}: Stage 3 {experiment} features are missing from dataset: {missing}"
                    )
                print(
                    f"[stage3] Dataset {dataset_index}/{total_datasets}: {dataset.name}, "
                    f"{experiment}.",
                    flush=True,
                )
                performance, windows = self._evaluate_feature_set(
                    dataset,
                    feature_columns,
                    experiment=experiment,
                    topk_percent=self._experiment_topk_percent(experiment),
                )
                rows.extend(performance)
                window_rows.extend(windows)

        result = pd.DataFrame(rows)
        time_windows = pd.DataFrame(window_rows)
        summary = self._summary(result)

        if output_dir is not None:
            ranking.to_csv(output_dir / "topk_global_feature_ranking.csv", index=False)
            selected_features.to_csv(output_dir / "stage3_selected_features.csv", index=False)
            selected_features.to_csv(output_dir / "topk_selected_features.csv", index=False)
            result.to_csv(output_dir / "topk_performance.csv", index=False)
            summary.to_csv(output_dir / "topk_performance_summary.csv", index=False)
            time_windows.to_csv(output_dir / "topk_time_windows.csv", index=False)
            self._write_dataset_metric_mean_table(summary, output_dir)
            self._write_topk_median_table(summary, output_dir)
            plot_stage3_topk_performance(summary, output_dir / "figures")
            print("[stage3] Top-k CSV and figures written.", flush=True)

        return result

    def _final_feature_ranking(
        self,
        feature_ranking: pd.DataFrame,
        datasets: list[PreparedDataset],
    ) -> pd.DataFrame:
        if "concept" not in feature_ranking.columns:
            raise ValueError("Stage 3 requires a Stage 2-B ranking table with a 'concept' column.")

        common_features = set(datasets[0].feature_columns)
        for dataset in datasets[1:]:
            common_features &= set(dataset.feature_columns)

        ranking = feature_ranking.copy()
        ranking = ranking[ranking["concept"].isin(common_features)].copy()
        if ranking.empty:
            raise ValueError("Stage 3 could not find any ranked raw features shared by all datasets.")
        if "meta_mdi_npsk_rank" not in ranking.columns:
            raise ValueError(
                "Stage 3 top-k selection requires the Stage 2-B 'meta_mdi_npsk_rank' column."
            )
        ranking["meta_mdi_npsk_rank"] = pd.to_numeric(
            ranking["meta_mdi_npsk_rank"],
            errors="raise",
        )
        if ranking["meta_mdi_npsk_rank"].isna().any():
            missing = ranking.loc[ranking["meta_mdi_npsk_rank"].isna(), "concept"].tolist()
            raise ValueError(f"Stage 3 found features without Stage 2-B NPSK ranks: {missing}")

        sort_columns = []
        ascending = []
        for column in ("meta_mdi_npsk_rank", "mean_rank", "concept"):
            if column in ranking.columns:
                sort_columns.append(column)
                ascending.append(True)
        ranking = ranking.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)
        ranking["final_feature_rank"] = range(1, len(ranking) + 1)
        ranking["feature"] = ranking["concept"]
        ranking["feature_family"] = ranking["feature"].map(family_label)
        ranking["meta_mdi_npsk_rank"] = ranking["meta_mdi_npsk_rank"].astype(int)

        preferred_columns = [
            "final_feature_rank",
            "feature",
            "feature_family",
            "meta_mdi_npsk_rank",
            "mean_rank",
            "rank1_frequency",
            "rank1_count",
            "dataset_count",
        ]
        columns = [column for column in preferred_columns if column in ranking.columns]
        return ranking[columns]

    def _selected_feature_sets(self, ranking: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for rank, feature in enumerate(KAMEI_FEATURE_COLUMNS, start=1):
            rows.append(
                {
                    "experiment": KAMEI_BASELINE_EXPERIMENT,
                    "topk_percent": pd.NA,
                    "selected_feature_count": len(KAMEI_FEATURE_COLUMNS),
                    "feature": feature,
                    "final_feature_rank": pd.NA,
                    "feature_family": "kamei",
                    "meta_mdi_npsk_rank": pd.NA,
                    "mean_rank": pd.NA,
                    "baseline_feature_rank": rank,
                    "selection_unit": "kamei_baseline",
                    "selected_rank_group_count": pd.NA,
                    "total_rank_group_count": pd.NA,
                    "rank_group_cutoff": pd.NA,
                }
            )

        rank_groups = sorted(ranking["meta_mdi_npsk_rank"].dropna().astype(int).unique().tolist())
        rank_group_count = len(rank_groups)
        if rank_group_count == 0:
            raise ValueError("Stage 3 could not find any Stage 2-B NPSK rank groups.")

        for topk_percent in self.config.topk_percentages:
            selected_rank_count = min(
                rank_group_count,
                max(1, ceil(rank_group_count * topk_percent / 100)),
            )
            selected_rank_groups = set(rank_groups[:selected_rank_count])
            rank_group_cutoff = max(selected_rank_groups)
            selected = ranking[ranking["meta_mdi_npsk_rank"].isin(selected_rank_groups)].copy()
            selected_count = len(selected)
            for _, record in selected.iterrows():
                rows.append(
                    {
                        "experiment": f"top_{topk_percent}",
                        "topk_percent": topk_percent,
                        "selected_feature_count": selected_count,
                        "feature": record["feature"],
                        "final_feature_rank": record["final_feature_rank"],
                        "feature_family": record.get("feature_family", family_label(record["feature"])),
                        "meta_mdi_npsk_rank": record.get("meta_mdi_npsk_rank", pd.NA),
                        "mean_rank": record.get("mean_rank", pd.NA),
                        "selection_unit": "npsk_rank_group",
                        "selected_rank_group_count": selected_rank_count,
                        "total_rank_group_count": rank_group_count,
                        "rank_group_cutoff": rank_group_cutoff,
                    }
                )
        return pd.DataFrame(rows)

    def _experiment_order(self) -> list[str]:
        return [
            KAMEI_BASELINE_EXPERIMENT,
            *[f"top_{topk_percent}" for topk_percent in self.config.topk_percentages],
        ]

    def _experiment_topk_percent(self, experiment: str) -> int | None:
        if experiment.startswith("top_"):
            return int(experiment.split("_", 1)[1])
        return None

    def _evaluate_feature_set(
        self,
        dataset: PreparedDataset,
        feature_columns: list[str],
        experiment: str,
        topk_percent: int | None,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        rows: list[dict[str, object]] = []
        window_rows: list[dict[str, object]] = []

        for window in self.splitter.split(dataset.frame):
            train = dataset.frame.iloc[window.train_start : window.train_end]
            test = dataset.frame.iloc[window.test_start : window.test_end]
            X_train = train[feature_columns]
            y_train = train[dataset.label_column].astype(int)
            X_test = test[feature_columns]
            y_test = test[dataset.label_column].astype(int)
            window_record = window.to_record(dataset.name, dataset.frame, dataset.date_column)
            window_record["experiment"] = experiment
            window_record["topk_percent"] = topk_percent
            window_record["selected_feature_count"] = len(feature_columns)
            window_record["used_for_modeling"] = True
            window_record["skip_reason"] = ""

            if y_train.nunique() < 2 or y_test.nunique() < 2:
                window_record["used_for_modeling"] = False
                window_record["skip_reason"] = "single_class_train_or_test"
                window_rows.append(window_record)
                print(
                    f"[stage3] {dataset.name}: {experiment}, "
                    f"window {window.window_id} skipped ({window_record['skip_reason']}).",
                    flush=True,
                )
                continue

            window_rows.append(window_record)

            experiment_seed = topk_percent if topk_percent is not None else 0
            seed = self.config.random_state + window.window_id + experiment_seed * 100
            params = self.rf_service.tune(X_train, y_train, random_state=seed)
            print(
                f"[stage3] {dataset.name}: {experiment}, "
                f"window {window.window_id} tuned params {params}.",
                flush=True,
            )

            for repeat in range(1, self.config.repeats + 1):
                run_seed = seed + repeat
                model = self.rf_service.fit(X_train, y_train, params=params, random_state=run_seed)
                metrics = self.rf_service.evaluate(model, X_test, y_test)
                selected_metrics = {metric: metrics[metric] for metric in STAGE3_METRIC_COLUMNS}
                rows.append(
                    {
                        **window_record,
                        "repeat": repeat,
                        **selected_metrics,
                    }
                )

        return rows, window_rows

    def _summary(self, result: pd.DataFrame) -> pd.DataFrame:
        if result.empty:
            return pd.DataFrame()
        available = [column for column in STAGE3_METRIC_COLUMNS if column in result.columns]
        summary = (
            result.groupby(["dataset", "experiment", "selected_feature_count"], as_index=False)[available]
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
        metric_mean_columns = [f"{metric}_mean" for metric in STAGE3_METRIC_COLUMNS]
        available = [column for column in metric_mean_columns if column in summary.columns]
        table = summary[["dataset", "experiment", "selected_feature_count", *available]].copy()
        table = table.rename(
            columns={f"{metric}_mean": metric for metric in STAGE3_METRIC_COLUMNS}
        )
        table.to_csv(output_dir / "stage3_topk_dataset_metric_means.csv", index=False)

        ordered_columns = [column for column in self._experiment_order() if column in set(table["experiment"])]
        for metric in STAGE3_METRIC_COLUMNS:
            if metric not in table.columns:
                continue
            matrix = table.pivot_table(
                index="dataset",
                columns="experiment",
                values=metric,
                aggfunc="mean",
            )
            matrix = matrix.reindex(columns=ordered_columns)
            matrix.to_csv(output_dir / f"stage3_{metric}_matrix.csv")

    def _write_topk_median_table(self, summary: pd.DataFrame, output_dir: Path) -> None:
        if summary.empty:
            return
        data = summary[summary["experiment"].astype(str).str.startswith("top_")].copy()
        if data.empty:
            return
        data["topk_percent"] = pd.to_numeric(
            data["experiment"].astype(str).str.replace("top_", "", regex=False),
            errors="coerce",
        )
        data = data.dropna(subset=["topk_percent"])
        if data.empty:
            return

        rows: list[dict[str, object]] = []
        for topk_percent, group in data.groupby("topk_percent"):
            row: dict[str, object] = {"topk_percent": int(topk_percent)}
            for metric in STAGE3_METRIC_COLUMNS:
                metric_column = f"{metric}_mean"
                if metric_column in group.columns:
                    row[f"{metric}_median_across_datasets"] = group[metric_column].median()
            rows.append(row)

        if not rows:
            return
        table = pd.DataFrame(rows).sort_values("topk_percent")
        table.to_csv(output_dir / "stage3_topk_median_across_datasets.csv", index=False)
