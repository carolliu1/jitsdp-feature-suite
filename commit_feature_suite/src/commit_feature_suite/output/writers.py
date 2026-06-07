"""Feature-level CSV output writers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


class FeatureOutputWriter:
    """Write function/file/commit feature tables."""

    def write_feature_tables(
        self,
        *,
        function_rows: List[Dict[str, Any]],
        file_metric_rows: List[Dict[str, Any]],
        commit_rows: List[Dict[str, Any]],
        output_prefix: Path,
    ) -> None:
        base_dir = output_prefix.parent
        base_dir.mkdir(parents=True, exist_ok=True)
        stem = output_prefix.stem
        function_output = base_dir / f"{stem}_function_level.csv"
        file_output = base_dir / f"{stem}_file_level.csv"
        commit_output = base_dir / f"{stem}_commit_level.csv"

        function_df = pd.DataFrame(function_rows)
        if function_df.empty:
            function_df = pd.DataFrame(columns=["commit_id", "method_id"])
        else:
            if "snapshot_scope" in function_df.columns:
                function_df = function_df[function_df["snapshot_scope"] == "current"].copy()
            if "method_id" in function_df.columns:
                function_df = function_df[function_df["method_id"].notna()].copy()
        function_df.to_csv(function_output, index=False, na_rep="None")

        metric_df = pd.DataFrame(file_metric_rows)
        if metric_df.empty:
            file_df = pd.DataFrame(columns=["commit_id", "metric_scope", "file_path"])
        else:
            if "snapshot_scope" in metric_df.columns:
                metric_df = metric_df[metric_df["snapshot_scope"] == "current"].copy()
            if "metric_scope" in metric_df.columns:
                file_df = metric_df[metric_df["metric_scope"] == "file"].copy()
            else:
                file_df = pd.DataFrame(columns=["commit_id", "metric_scope", "file_path"])
        file_df.to_csv(file_output, index=False, na_rep="None")

        commit_df = pd.DataFrame(commit_rows)
        if commit_df.empty:
            commit_df = pd.DataFrame(columns=["commit_id", "commit_author_date"])
        commit_df.to_csv(commit_output, index=False, na_rep="None")
