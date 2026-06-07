"""Analysis result models used by the pipeline orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class CommitAnalysisResult:
    """Rows produced for a single commit analysis pass."""

    function_rows: List[Dict[str, Any]] = field(default_factory=list)
    file_metric_rows: List[Dict[str, Any]] = field(default_factory=list)
    commit_rows: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """Accumulator for all commit-level results in one run."""

    function_rows: List[Dict[str, Any]] = field(default_factory=list)
    file_metric_rows: List[Dict[str, Any]] = field(default_factory=list)
    commit_rows: List[Dict[str, Any]] = field(default_factory=list)

    def extend(self, item: CommitAnalysisResult) -> None:
        self.function_rows.extend(item.function_rows)
        self.file_metric_rows.extend(item.file_metric_rows)
        self.commit_rows.extend(item.commit_rows)
