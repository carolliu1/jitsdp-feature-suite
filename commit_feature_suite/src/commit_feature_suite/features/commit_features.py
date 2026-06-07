"""Commit-level feature builders and aggregators."""

from __future__ import annotations

from typing import Any, Dict, List

from commit_feature_suite.features.aggregation import append_if_number, stats_prefix
from commit_feature_suite.metrics.coupling import MethodCouplingRecord


class CommitFeatureBuilder:
    def __init__(self, method_agg_metrics: List[str], file_agg_metrics: List[str]) -> None:
        self.method_agg_metrics = method_agg_metrics
        self.file_agg_metrics = file_agg_metrics

    def init_scope_buffers(self) -> tuple[Dict[str, Dict[str, List[float]]], Dict[str, Dict[str, List[float]]], Dict[str, set[str]], Dict[str, int]]:
        scoped_method_metrics: Dict[str, Dict[str, List[float]]] = {
            "current": {name: [] for name in self.method_agg_metrics},
        }
        scoped_file_metrics: Dict[str, Dict[str, List[float]]] = {
            "current": {name: [] for name in self.file_agg_metrics},
        }
        scoped_modified_method_ids: Dict[str, set[str]] = {"current": set()}
        class_touched_count_by_scope: Dict[str, int] = {"current": 0}
        return scoped_method_metrics, scoped_file_metrics, scoped_modified_method_ids, class_touched_count_by_scope

    def accumulate_scope_metrics(
        self,
        *,
        scope: str,
        affected_methods,
        method_metric_map: Dict[str, Any],
        method_coupling_map: Dict[str, MethodCouplingRecord],
        global_var_count_map: Dict[str, int],
        file_metric_map: Dict[str, Any],
        scoped_method_metrics: Dict[str, Dict[str, List[float]]],
        scoped_file_metrics: Dict[str, Dict[str, List[float]]],
    ) -> None:
        for item in affected_methods:
            method = item.method
            method_metric = method_metric_map.get(method.method_id)
            method_coupling = method_coupling_map.get(method.method_id)
            if method_coupling is not None:
                append_if_number(scoped_method_metrics[scope]["method_in_coupling"], method_coupling.in_coupling)
                append_if_number(scoped_method_metrics[scope]["method_out_coupling"], method_coupling.out_coupling)
            append_if_number(scoped_method_metrics[scope]["method_global_var_count"], global_var_count_map.get(method.method_id, 0))
            if method_metric is None:
                continue
            append_if_number(scoped_method_metrics[scope]["method_cc"], method_metric.cc)
            append_if_number(scoped_method_metrics[scope]["method_halstead"], method_metric.halstead)
            append_if_number(scoped_method_metrics[scope]["method_halstead_n1"], method_metric.halstead_n1)
            append_if_number(scoped_method_metrics[scope]["method_halstead_n2"], method_metric.halstead_n2)
            append_if_number(scoped_method_metrics[scope]["method_halstead_N1"], method_metric.halstead_N1)
            append_if_number(scoped_method_metrics[scope]["method_halstead_N2"], method_metric.halstead_N2)
            append_if_number(scoped_method_metrics[scope]["method_halstead_length"], method_metric.halstead_length)
            append_if_number(scoped_method_metrics[scope]["method_halstead_vocabulary"], method_metric.halstead_vocabulary)
            append_if_number(scoped_method_metrics[scope]["method_halstead_volume"], method_metric.halstead_volume)
            append_if_number(scoped_method_metrics[scope]["method_halstead_difficulty"], method_metric.halstead_difficulty)
            append_if_number(scoped_method_metrics[scope]["method_halstead_effort"], method_metric.halstead_effort)
            append_if_number(scoped_method_metrics[scope]["method_halstead_bugs"], method_metric.halstead_bugs)
            append_if_number(scoped_method_metrics[scope]["method_halstead_time"], method_metric.halstead_time)
            append_if_number(scoped_method_metrics[scope]["method_nargs"], method_metric.nargs)
            append_if_number(scoped_method_metrics[scope]["method_nexits"], method_metric.nexits)

        for metric in file_metric_map.values():
            append_if_number(scoped_file_metrics[scope]["file_cloc"], metric.cloc)
            append_if_number(scoped_file_metrics[scope]["file_mi"], metric.mi)
            append_if_number(scoped_file_metrics[scope]["file_nom"], metric.nom)
            append_if_number(scoped_file_metrics[scope]["file_class"], metric.file_class)

    @staticmethod
    def current_new_file_stats(modified_files) -> tuple[int, float]:
        modified_files = list(modified_files or [])
        if not modified_files:
            return 0, 0.0
        new_count = 0
        for modified_file in modified_files:
            change_type = getattr(modified_file, "change_type", None)
            change_name = str(getattr(change_type, "name", "")).upper() if change_type is not None else ""
            if change_name == "ADD":
                new_count += 1
                continue
            old_path = getattr(modified_file, "old_path", None)
            new_path = getattr(modified_file, "new_path", None)
            if old_path is None and new_path is not None:
                new_count += 1
        ratio = float(new_count) / float(len(modified_files))
        return new_count, ratio

    def build_commit_feature_row(
        self,
        *,
        commit,
        scoped_method_metrics: Dict[str, Dict[str, List[float]]],
        scoped_file_metrics: Dict[str, Dict[str, List[float]]],
        scoped_modified_method_ids: Dict[str, set[str]],
        current_new_file_count: int,
        current_new_file_ratio: float,
        class_touched_count_by_scope: Dict[str, int],
    ) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "commit_id": commit.hash,
            "commit_author_date": getattr(commit, "author_date", None),
        }
        for scope in ("current",):
            for metric_name in self.method_agg_metrics:
                row.update(stats_prefix(scoped_method_metrics[scope][metric_name], f"{scope}_{metric_name}"))
            for metric_name in self.file_agg_metrics:
                row.update(stats_prefix(scoped_file_metrics[scope][metric_name], f"{scope}_{metric_name}"))

        current_count = len(scoped_modified_method_ids["current"])
        row["modified_method_count_current"] = current_count
        row["has_modified_methods_current"] = current_count > 0
        row["current_new_file_count"] = current_new_file_count
        row["current_new_file_ratio"] = current_new_file_ratio
        row["class_touched_count_current"] = class_touched_count_by_scope.get("current", 0)
        row["is_complete_feature"] = self._is_feature_extraction_complete_current(row)
        return row

    @staticmethod
    def _is_feature_extraction_complete_current(row: Dict[str, Any]) -> bool:
        """Whether current-scope commit features are fully extracted.

        Rule:
        - If any current-scope feature value is None, treat as incomplete.
        - Otherwise treat as complete.
        """
        for key, value in row.items():
            if key.startswith("current_") and value is None:
                return False
        return True

    @staticmethod
    def build_commit_stats_row(*, commit, current_in_values: List[int], current_out_values: List[int]) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "commit_id": commit.hash,
            "commit_author_date": getattr(commit, "author_date", None),
        }
        row.update(stats_prefix(current_in_values, "current_in"))
        row.update(stats_prefix(current_out_values, "current_out"))
        return row
