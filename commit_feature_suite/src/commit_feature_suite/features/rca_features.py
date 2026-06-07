"""RCA metric row builders."""

from __future__ import annotations

from typing import Any, Dict


class RCAMetricRowBuilder:
    @staticmethod
    def build_metric_rows(
        *,
        commit,
        snapshot_scope: str,
        snapshot_hash: str,
        method_metric_map: Dict[str, Any],
        file_metric_map: Dict[str, Any],
        method_info_map: Dict[str, Any],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for method_id, metric in method_metric_map.items():
            method = method_info_map.get(method_id)
            rows.append(
                {
                    "commit_id": commit.hash,
                    "commit_author_date": getattr(commit, "author_date", None),
                    "snapshot_scope": snapshot_scope,
                    "snapshot_commit_id": snapshot_hash,
                    "metric_scope": "method",
                    "method_id": method_id,
                    "file_path": method.file_path if method else None,
                    "class_name": method.class_name if method else None,
                    "method_name": method.method_name if method else None,
                    "start_line": method.start_line if method else None,
                    "end_line": method.end_line if method else None,
                    "cc": metric.cc,
                    "halstead": metric.halstead,
                    "halstead_n1": metric.halstead_n1,
                    "halstead_n2": metric.halstead_n2,
                    "halstead_N1": metric.halstead_N1,
                    "halstead_N2": metric.halstead_N2,
                    "halstead_length": metric.halstead_length,
                    "halstead_vocabulary": metric.halstead_vocabulary,
                    "halstead_volume": metric.halstead_volume,
                    "halstead_difficulty": metric.halstead_difficulty,
                    "halstead_effort": metric.halstead_effort,
                    "halstead_bugs": metric.halstead_bugs,
                    "halstead_time": metric.halstead_time,
                    "nargs": metric.nargs,
                    "nexits": metric.nexits,
                    "cloc": None,
                    "mi": None,
                    "nom": None,
                    "file_class": None,
                }
            )
        for file_path, metric in file_metric_map.items():
            rows.append(
                {
                    "commit_id": commit.hash,
                    "commit_author_date": getattr(commit, "author_date", None),
                    "snapshot_scope": snapshot_scope,
                    "snapshot_commit_id": snapshot_hash,
                    "metric_scope": "file",
                    "method_id": None,
                    "file_path": file_path,
                    "class_name": None,
                    "method_name": None,
                    "start_line": None,
                    "end_line": None,
                    "cc": None,
                    "halstead": None,
                    "halstead_n1": None,
                    "halstead_n2": None,
                    "halstead_N1": None,
                    "halstead_N2": None,
                    "halstead_length": None,
                    "halstead_vocabulary": None,
                    "halstead_volume": None,
                    "halstead_difficulty": None,
                    "halstead_effort": None,
                    "halstead_bugs": None,
                    "halstead_time": None,
                    "nargs": None,
                    "nexits": None,
                    "cloc": metric.cloc,
                    "mi": metric.mi,
                    "nom": metric.nom,
                    "file_class": metric.file_class,
                }
            )
        return rows

