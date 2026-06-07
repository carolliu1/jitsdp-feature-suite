from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import Stage2Config
from .data import PreparedDataset
from .feature_groups import FeatureGrouper
from .modeling import RandomForestService, TimeForwardSplitter
from .npsk import ScottKnottESD
from .plots import (
    family_label,
    plot_stage2_npsk_ranks,
    plot_stage2_npsk_ranks_long,
    plot_stage2_rank_matrix,
    plot_stage2_top_rank_frequency,
)


@dataclass
class Stage2Result:
    raw_mdi: dict[str, pd.DataFrame]
    grouped_mdi: dict[str, pd.DataFrame]
    stage2a_local_rankings: dict[str, pd.DataFrame]
    stage2a_rank_matrix: pd.DataFrame
    stage2a_top_rank_frequency: pd.DataFrame
    stage2a_meta_npsk_ranking: pd.DataFrame
    stage2a_mdi_npsk_rank_table: pd.DataFrame
    stage2a_mdi_npsk_rank_long: pd.DataFrame
    stage2b_local_rankings: dict[str, pd.DataFrame]
    stage2b_rank_matrix: pd.DataFrame
    stage2b_top_rank_frequency: pd.DataFrame
    stage2b_meta_npsk_ranking: pd.DataFrame
    stage2b_mdi_npsk_rank_table: pd.DataFrame
    stage2b_mdi_npsk_rank_long: pd.DataFrame
    local_rankings: dict[str, pd.DataFrame]
    rank_matrix: pd.DataFrame
    top_rank_frequency: pd.DataFrame
    meta_npsk_ranking: pd.DataFrame
    performance: pd.DataFrame
    time_windows: pd.DataFrame
    tuned_params: pd.DataFrame


class Stage2MDIAnalyzer:
    """Time-forward RF/MDI analysis with grouped MDI and local/meta NPSK."""

    def __init__(self, config: Stage2Config | None = None) -> None:
        self.config = config or Stage2Config()
        self.splitter = TimeForwardSplitter(self.config.time_windows)
        self.rf_service = RandomForestService(self.config.random_forest)

    def run(
        self,
        datasets: list[PreparedDataset],
        output_dir: Path | None = None,
    ) -> Stage2Result:
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)

        grouped_mdi: dict[str, pd.DataFrame] = {}
        raw_mdi: dict[str, pd.DataFrame] = {}
        performance_rows: list[dict[str, object]] = []
        time_window_rows: list[dict[str, object]] = []
        tuned_param_rows: list[dict[str, object]] = []

        total_datasets = len(datasets)
        for dataset_index, dataset in enumerate(datasets, start=1):
            print(
                f"[stage2] Dataset {dataset_index}/{total_datasets}: {dataset.name} "
                f"({len(dataset.frame)} rows, {len(dataset.feature_columns)} features)",
                flush=True,
            )
            raw_matrix, grouped_matrix, performance, windows, tuned_params = self._run_one_dataset(
                dataset,
            )
            raw_mdi[dataset.name] = raw_matrix
            grouped_mdi[dataset.name] = grouped_matrix
            performance_rows.extend(performance)
            time_window_rows.extend(windows)
            tuned_param_rows.extend(tuned_params)

        performance_frame = pd.DataFrame(performance_rows)
        time_windows_frame = pd.DataFrame(time_window_rows)
        tuned_params_frame = pd.DataFrame(tuned_param_rows)

        if output_dir is not None:
            performance_frame.to_csv(output_dir / "window_performance.csv", index=False)
            time_windows_frame.to_csv(output_dir / "time_windows.csv", index=False)
            tuned_params_frame.to_csv(output_dir / "tuned_params.csv", index=False)

        print("[stage2-A] Ranking grouped MDI concepts with local/meta NPSK...", flush=True)
        (
            stage2a_local_rankings,
            stage2a_rank_matrix,
            stage2a_top_rank_frequency,
            stage2a_meta_npsk,
            stage2a_rank_table,
            stage2a_rank_long,
        ) = self._rank_mdi_scheme(
            mdi_matrices=grouped_mdi,
            output_dir=output_dir / "stage2A_grouped_mdi" if output_dir is not None else None,
            scheme_label="Stage 2-A Grouped MDI",
            entity_label="Concept",
            mdi_file_suffix="grouped_mdi",
            rank_table_stem="all_concept_mdi_npsk_ranks",
        )

        print("[stage2-B] Ranking raw feature MDI with local/meta NPSK...", flush=True)
        (
            stage2b_local_rankings,
            stage2b_rank_matrix,
            stage2b_top_rank_frequency,
            stage2b_meta_npsk,
            stage2b_rank_table,
            stage2b_rank_long,
        ) = self._rank_mdi_scheme(
            mdi_matrices=raw_mdi,
            output_dir=output_dir / "stage2B_raw_feature_mdi" if output_dir is not None else None,
            scheme_label="Stage 2-B Raw Feature MDI",
            entity_label="Feature",
            mdi_file_suffix="raw_feature_mdi",
            rank_table_stem="all_feature_mdi_npsk_ranks",
        )

        return Stage2Result(
            raw_mdi,
            grouped_mdi,
            stage2a_local_rankings,
            stage2a_rank_matrix,
            stage2a_top_rank_frequency,
            stage2a_meta_npsk,
            stage2a_rank_table,
            stage2a_rank_long,
            stage2b_local_rankings,
            stage2b_rank_matrix,
            stage2b_top_rank_frequency,
            stage2b_meta_npsk,
            stage2b_rank_table,
            stage2b_rank_long,
            stage2a_local_rankings,
            stage2a_rank_matrix,
            stage2a_top_rank_frequency,
            stage2a_meta_npsk,
            performance_frame,
            time_windows_frame,
            tuned_params_frame,
        )

    def _rank_mdi_scheme(
        self,
        mdi_matrices: dict[str, pd.DataFrame],
        output_dir: Path | None,
        scheme_label: str,
        entity_label: str,
        mdi_file_suffix: str,
        rank_table_stem: str,
    ) -> tuple[
        dict[str, pd.DataFrame],
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)

        local_rankings: dict[str, pd.DataFrame] = {}
        for dataset_name, matrix in mdi_matrices.items():
            print(f"[{scheme_label}] {dataset_name}: computing local NPSK-ESD ranks...", flush=True)
            ranking_input = matrix.drop(columns=["dataset", "window_id", "bootstrap_id"], errors="ignore")
            local_ranking = ScottKnottESD(self.config.npsk).rank(ranking_input, higher_is_better=True)
            local_ranking["dataset"] = dataset_name
            local_rankings[dataset_name] = local_ranking

            if output_dir is not None:
                matrix.to_csv(output_dir / f"{dataset_name}_{mdi_file_suffix}.csv", index=False)
                local_ranking.to_csv(output_dir / f"{dataset_name}_local_npsk_ranks.csv", index=False)
                figure_dir = output_dir / "figures"
                plot_stage2_npsk_ranks(
                    local_ranking,
                    figure_dir,
                    f"{dataset_name}_local_npsk_ranks",
                    f"{scheme_label}: {dataset_name} Local NPSK Ranks for All {entity_label}s",
                )

        print(f"[{scheme_label}] Building cross-dataset rank matrix and meta-NPSK...", flush=True)
        rank_matrix = self._build_rank_matrix(local_rankings)
        top_rank_frequency = self._top_rank_frequency(rank_matrix)
        meta_npsk = ScottKnottESD(self.config.npsk).rank(rank_matrix, higher_is_better=False)
        mdi_npsk_rank_table = self._mdi_npsk_rank_table(rank_matrix, meta_npsk, top_rank_frequency)
        mdi_npsk_rank_long = self._mdi_npsk_rank_long(local_rankings, meta_npsk)

        if output_dir is not None:
            rank_matrix.to_csv(output_dir / "rank_matrix.csv")
            top_rank_frequency.to_csv(output_dir / "top_rank_frequency.csv", index=False)
            meta_npsk.to_csv(output_dir / "meta_npsk_ranks.csv", index=False)
            mdi_npsk_rank_table.to_csv(output_dir / f"{rank_table_stem}.csv", index=False)
            mdi_npsk_rank_long.to_csv(output_dir / f"{rank_table_stem}_long.csv", index=False)
            figure_dir = output_dir / "figures"
            plot_stage2_top_rank_frequency(top_rank_frequency, figure_dir)
            plot_stage2_npsk_ranks(
                meta_npsk,
                figure_dir,
                "meta_npsk_ranks",
                f"{scheme_label}: Meta-NPSK Ranks for All {entity_label}s",
            )
            plot_stage2_npsk_ranks(
                mdi_npsk_rank_table.rename(columns={"meta_mdi_npsk_rank": "rank"}),
                figure_dir,
                rank_table_stem,
                f"{scheme_label}: Final MDI NPSK Ranks for All {entity_label}s",
            )
            plot_stage2_npsk_ranks_long(
                mdi_npsk_rank_long,
                figure_dir,
                f"{rank_table_stem}_long",
                f"{scheme_label}: Local and Meta NPSK Ranks for All {entity_label}s",
            )
            plot_stage2_rank_matrix(rank_matrix, figure_dir)
            print(f"[{scheme_label}] CSV and figures written.", flush=True)

        return (
            local_rankings,
            rank_matrix,
            top_rank_frequency,
            meta_npsk,
            mdi_npsk_rank_table,
            mdi_npsk_rank_long,
        )

    def _run_one_dataset(
        self,
        dataset: PreparedDataset,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        list[dict[str, object]],
        list[dict[str, object]],
        list[dict[str, object]],
    ]:
        feature_columns = dataset.feature_columns

        if not feature_columns:
            raise ValueError(f"{dataset.name}: no feature columns selected for Stage 2.")

        selected_grouper = FeatureGrouper(feature_columns)
        raw_rows: list[pd.Series] = []
        grouped_rows: list[pd.Series] = []
        performance_rows: list[dict[str, object]] = []
        time_window_rows: list[dict[str, object]] = []
        tuned_param_rows: list[dict[str, object]] = []

        windows = list(self.splitter.split(dataset.frame))
        print(f"[stage2] {dataset.name}: generated {len(windows)} time-forward windows.", flush=True)
        for window_index, window in enumerate(windows, start=1):
            print(f"[stage2] {dataset.name}: window {window_index}/{len(windows)} tuning RF...", flush=True)
            train = dataset.frame.iloc[window.train_start : window.train_end]
            test = dataset.frame.iloc[window.test_start : window.test_end]
            X_train = train[feature_columns]
            y_train = train[dataset.label_column].astype(int)
            X_test = test[feature_columns]
            y_test = test[dataset.label_column].astype(int)
            window_record = window.to_record(dataset.name, dataset.frame, dataset.date_column)
            window_record["used_for_modeling"] = True
            window_record["skip_reason"] = ""

            if y_train.nunique() < 2 or y_test.nunique() < 2:
                window_record["used_for_modeling"] = False
                window_record["skip_reason"] = "single_class_train_or_test"
                time_window_rows.append(window_record)
                skip_reason = window_record["skip_reason"]
                print(f"[stage2] {dataset.name}: window {window.window_id} skipped ({skip_reason}).", flush=True)
                continue

            time_window_rows.append(window_record)

            seed = self.config.random_state + window.window_id
            params = self.rf_service.tune(X_train, y_train, random_state=seed)
            tuned_param_rows.append(
                {
                    "dataset": dataset.name,
                    "window_id": window.window_id,
                    **params,
                }
            )

            print(f"[stage2] {dataset.name}: window {window.window_id} tuned params {params}.", flush=True)
            for bootstrap_id in range(1, self.config.bootstrap_iterations + 1):
                if bootstrap_id == 1 or bootstrap_id == self.config.bootstrap_iterations or bootstrap_id % 5 == 0:
                    print(
                        f"[stage2] {dataset.name}: window {window.window_id} "
                        f"bootstrap {bootstrap_id}/{self.config.bootstrap_iterations}",
                        flush=True,
                    )
                run_seed = seed * 1000 + bootstrap_id
                X_resampled, y_resampled = self.rf_service.bootstrap_sample(X_train, y_train, random_state=run_seed)
                model = self.rf_service.fit(X_resampled, y_resampled, params=params, random_state=run_seed)
                raw_importances = self.rf_service.feature_importances(model, feature_columns)
                grouped_importances = selected_grouper.sum_importances(raw_importances)

                raw_importances_with_ids = raw_importances.copy()
                raw_importances_with_ids["dataset"] = dataset.name
                raw_importances_with_ids["window_id"] = window.window_id
                raw_importances_with_ids["bootstrap_id"] = bootstrap_id
                raw_rows.append(raw_importances_with_ids)

                grouped_importances["dataset"] = dataset.name
                grouped_importances["window_id"] = window.window_id
                grouped_importances["bootstrap_id"] = bootstrap_id
                grouped_rows.append(grouped_importances)

                metrics = self.rf_service.evaluate(model, X_test, y_test)
                performance_rows.append(
                    {
                        "dataset": dataset.name,
                        "window_id": window.window_id,
                        "bootstrap_id": bootstrap_id,
                        **metrics,
                    }
                )

        if not grouped_rows:
            raise ValueError(f"{dataset.name}: Stage 2 produced no valid time windows.")

        raw_matrix = pd.DataFrame(raw_rows)
        grouped_matrix = pd.DataFrame(grouped_rows)
        id_columns = ["dataset", "window_id", "bootstrap_id"]
        raw_value_columns = sorted(column for column in raw_matrix.columns if column not in id_columns)
        grouped_value_columns = sorted(column for column in grouped_matrix.columns if column not in id_columns)
        return (
            raw_matrix[id_columns + raw_value_columns],
            grouped_matrix[id_columns + grouped_value_columns],
            performance_rows,
            time_window_rows,
            tuned_param_rows,
        )

    def _build_rank_matrix(self, local_rankings: dict[str, pd.DataFrame]) -> pd.DataFrame:
        rows = []
        for dataset_name, ranking in local_rankings.items():
            row = ranking.set_index("concept")["rank"]
            row.name = dataset_name
            rows.append(row)
        matrix = pd.DataFrame(rows)
        matrix.index.name = "dataset"
        return matrix

    def _mdi_npsk_rank_table(
        self,
        rank_matrix: pd.DataFrame,
        meta_npsk: pd.DataFrame,
        top_rank_frequency: pd.DataFrame,
    ) -> pd.DataFrame:
        table = rank_matrix.T.reset_index().rename(columns={"index": "concept"})
        table = table.rename(columns={column: f"{column}_local_mdi_rank" for column in rank_matrix.index})

        meta_columns = ["concept", "rank", "cluster", "n"]
        available_meta_columns = [column for column in meta_columns if column in meta_npsk.columns]
        meta = meta_npsk[available_meta_columns].rename(
            columns={
                "rank": "meta_mdi_npsk_rank",
                "cluster": "meta_mdi_npsk_cluster",
                "n": "meta_mdi_npsk_n",
            }
        )
        table = table.merge(meta, on="concept", how="left")

        frequency_columns = ["concept", "rank1_count", "rank1_frequency", "mean_rank", "dataset_count"]
        available_frequency_columns = [column for column in frequency_columns if column in top_rank_frequency.columns]
        table = table.merge(top_rank_frequency[available_frequency_columns], on="concept", how="left")
        table["feature_family"] = table["concept"].map(family_label)
        return table.sort_values(["meta_mdi_npsk_rank", "mean_rank", "concept"]).reset_index(drop=True)

    def _mdi_npsk_rank_long(
        self,
        local_rankings: dict[str, pd.DataFrame],
        meta_npsk: pd.DataFrame,
    ) -> pd.DataFrame:
        rows = []
        for dataset_name, ranking in local_rankings.items():
            for _, record in ranking.iterrows():
                rows.append(
                    {
                        "rank_scope": "local_dataset",
                        "dataset": dataset_name,
                        "concept": record["concept"],
                        "rank": record["rank"],
                        "cluster": record.get("cluster", pd.NA),
                        "n": record.get("n", pd.NA),
                        "feature_family": family_label(record["concept"]),
                    }
                )
        for _, record in meta_npsk.iterrows():
            rows.append(
                {
                    "rank_scope": "meta_dataset",
                    "dataset": "ALL_DATASETS",
                    "concept": record["concept"],
                    "rank": record["rank"],
                    "cluster": record.get("cluster", pd.NA),
                    "n": record.get("n", pd.NA),
                    "feature_family": family_label(record["concept"]),
                }
            )
        return pd.DataFrame(rows)

    def _top_rank_frequency(self, rank_matrix: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for concept in rank_matrix.columns:
            values = rank_matrix[concept].dropna()
            rows.append(
                {
                    "concept": concept,
                    "feature_family": family_label(concept),
                    "dataset_count": int(values.shape[0]),
                    "rank1_count": int((values == 1).sum()),
                    "rank1_frequency": float((values == 1).mean()) if not values.empty else 0.0,
                    "mean_rank": float(values.mean()) if not values.empty else pd.NA,
                }
            )
        return pd.DataFrame(rows).sort_values(["rank1_frequency", "mean_rank"], ascending=[False, True]).reset_index(drop=True)
