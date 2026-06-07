"""Function-level feature row builders."""

from __future__ import annotations

from typing import Any, Dict

from commit_feature_suite.gitops.commits import is_merge_commit
from commit_feature_suite.graph.result import CallGraphBuildResult
from commit_feature_suite.metrics.coupling import MethodCouplingRecord
from commit_feature_suite.metrics.rust_code_analysis import FileMetricRecord, MethodMetricRecord


class FunctionFeatureBuilder:
    """Build function-level rows from affected methods and metrics."""

    @staticmethod
    def build_output_row(
        *,
        commit,
        graph_result: CallGraphBuildResult,
        affected_method,
        snapshot_scope: str,
        snapshot_hash: str,
        method_metric_map: Dict[str, MethodMetricRecord],
        file_metric_map: Dict[str, FileMetricRecord],
        global_var_count_map: Dict[str, int],
        method_coupling_map: Dict[str, MethodCouplingRecord],
    ) -> Dict[str, Any]:
        method = affected_method.method
        old_path = affected_method.old_path
        new_path = affected_method.new_path
        file_path_for_metric = old_path if snapshot_scope == "parent" else (new_path or old_path)

        method_metric = method_metric_map.get(method.method_id)
        file_metric = file_metric_map.get(file_path_for_metric or "")
        method_coupling = method_coupling_map.get(method.method_id)
        return {
            "commit_id": commit.hash,
            "commit_author_date": getattr(commit, "author_date", None),
            "snapshot_scope": snapshot_scope,
            "snapshot_commit_id": snapshot_hash,
            "is_merge_commit": bool(is_merge_commit(commit)),
            "method_coupling_available": True,
            "method_id": method.method_id,
            "file_path": method.file_path,
            "class_name": method.class_name,
            "method_name": method.method_name,
            "start_line": method.start_line,
            "end_line": method.end_line,
            "method_in_coupling": method_coupling.in_coupling if method_coupling else None,
            "method_out_coupling": method_coupling.out_coupling if method_coupling else None,
            "old_path": old_path,
            "new_path": new_path,
            "language": method.language,
            "node_count": graph_result.node_count,
            "file_count": graph_result.file_count,
            "token_count": getattr(method, "token_count", None),
            "method_cc": method_metric.cc if method_metric else None,
            "method_halstead": method_metric.halstead if method_metric else None,
            "method_halstead_n1": method_metric.halstead_n1 if method_metric else None,
            "method_halstead_n2": method_metric.halstead_n2 if method_metric else None,
            "method_halstead_N1": method_metric.halstead_N1 if method_metric else None,
            "method_halstead_N2": method_metric.halstead_N2 if method_metric else None,
            "method_halstead_length": method_metric.halstead_length if method_metric else None,
            "method_halstead_vocabulary": method_metric.halstead_vocabulary if method_metric else None,
            "method_halstead_volume": method_metric.halstead_volume if method_metric else None,
            "method_halstead_difficulty": method_metric.halstead_difficulty if method_metric else None,
            "method_halstead_effort": method_metric.halstead_effort if method_metric else None,
            "method_halstead_bugs": method_metric.halstead_bugs if method_metric else None,
            "method_halstead_time": method_metric.halstead_time if method_metric else None,
            "method_nargs": method_metric.nargs if method_metric else None,
            "method_nexits": method_metric.nexits if method_metric else None,
            "method_global_var_count": global_var_count_map.get(method.method_id, 0),
            "file_cloc": file_metric.cloc if file_metric else None,
            "file_mi": file_metric.mi if file_metric else None,
            "file_nom": file_metric.nom if file_metric else None,
            "file_class": file_metric.file_class if file_metric else None,
        }

    @classmethod
    def build_non_method_coupling_row(
        cls,
        *,
        commit,
        graph_result: CallGraphBuildResult,
        modified_file,
        snapshot_scope: str,
        snapshot_hash: str,
        file_metric_map: Dict[str, FileMetricRecord],
    ) -> Dict[str, Any]:
        old_path = getattr(modified_file, "old_path", None)
        new_path = getattr(modified_file, "new_path", None)
        language = getattr(modified_file, "language_supported", None) or ""
        path_for_snapshot = (old_path or new_path) if snapshot_scope == "parent" else (new_path or old_path)
        file_metric = file_metric_map.get(path_for_snapshot or "")
        return {
            "commit_id": commit.hash,
            "commit_author_date": getattr(commit, "author_date", None),
            "snapshot_scope": snapshot_scope,
            "snapshot_commit_id": snapshot_hash,
            "is_merge_commit": bool(is_merge_commit(commit)),
            "method_coupling_available": False,
            "method_id": None,
            "file_path": old_path or new_path,
            "class_name": None,
            "method_name": None,
            "start_line": None,
            "end_line": None,
            "method_in_coupling": None,
            "method_out_coupling": None,
            "old_path": old_path,
            "new_path": new_path,
            "language": language,
            "node_count": graph_result.node_count,
            "file_count": graph_result.file_count,
            "token_count": cls.token_count_from_modified_file(modified_file),
            "method_cc": None,
            "method_halstead": None,
            "method_halstead_n1": None,
            "method_halstead_n2": None,
            "method_halstead_N1": None,
            "method_halstead_N2": None,
            "method_halstead_length": None,
            "method_halstead_vocabulary": None,
            "method_halstead_volume": None,
            "method_halstead_difficulty": None,
            "method_halstead_effort": None,
            "method_halstead_bugs": None,
            "method_halstead_time": None,
            "method_nargs": None,
            "method_nexits": None,
            "method_global_var_count": None,
            "file_cloc": file_metric.cloc if file_metric else None,
            "file_mi": file_metric.mi if file_metric else None,
            "file_nom": file_metric.nom if file_metric else None,
            "file_class": file_metric.file_class if file_metric else None,
        }

    @staticmethod
    def token_count_from_modified_file(modified_file) -> int:
        try:
            source_code_before = getattr(modified_file, "source_code_before", None) or ""
        except Exception:
            return 0
        if not source_code_before:
            return 0
        return len(source_code_before.split())
