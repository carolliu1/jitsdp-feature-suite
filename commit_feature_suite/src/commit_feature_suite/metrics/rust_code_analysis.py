"""rust-code-analysis metrics extraction (method-level + file-level)."""

from __future__ import annotations

import json
import hashlib
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

from commit_feature_suite.models import MethodInfo


@dataclass
class MethodMetricRecord:
    """Method-level metrics from rust-code-analysis."""

    method_id: str
    cc: float | None
    halstead: float | None
    halstead_n1: float | None
    halstead_n2: float | None
    halstead_N1: float | None
    halstead_N2: float | None
    halstead_length: float | None
    halstead_vocabulary: float | None
    halstead_volume: float | None
    halstead_difficulty: float | None
    halstead_effort: float | None
    halstead_bugs: float | None
    halstead_time: float | None
    nargs: float | None
    nexits: float | None


@dataclass
class FileMetricRecord:
    """File-level metrics from rust-code-analysis."""

    file_path: str
    cloc: float | None
    mi: float | None
    nom: float | None
    file_class: float | None = None


class RustCodeAnalysisExtractor:
    """Extract metrics with rust-code-analysis for selected methods/files only."""

    def __init__(self, command: str, logger, timeout_seconds: int = 120, debug_dump_dir: Path | None = None) -> None:
        self.command = command
        self.logger = logger
        self.timeout_seconds = timeout_seconds
        self.debug_dump_dir = debug_dump_dir
        if self.debug_dump_dir is not None:
            self.debug_dump_dir.mkdir(parents=True, exist_ok=True)

    def extract_for_modified_file(
        self,
        *,
        logical_file_path: str,
        source_code: str,
        affected_methods_in_file: Iterable[MethodInfo],
    ) -> tuple[FileMetricRecord, Dict[str, MethodMetricRecord]]:
        method_list = list(affected_methods_in_file)
        suffix = Path(logical_file_path).suffix or ".txt"
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=suffix, delete=True) as temp_file:
            temp_file.write(source_code)
            temp_file.flush()
            parsed = self._run_and_parse(Path(temp_file.name))
        self._dump_debug_json(logical_file_path, parsed)

        file_metrics = FileMetricRecord(
            file_path=logical_file_path,
            cloc=self._extract_file_metric(parsed, {"cloc", "comment_lines"}),
            mi=self._extract_file_metric(parsed, {"mi", "maintainability_index"}),
            nom=self._extract_file_metric(parsed, {"nom", "functions", "methods_count"}),
        )

        method_metrics: Dict[str, MethodMetricRecord] = {}
        method_spaces = self._collect_method_spaces(parsed)
        for method in method_list:
            matched = self._best_match_method_space(method, method_spaces)
            if matched is None:
                method_metrics[method.method_id] = MethodMetricRecord(
                    method_id=method.method_id,
                    cc=None,
                    halstead=None,
                    halstead_n1=None,
                    halstead_n2=None,
                    halstead_N1=None,
                    halstead_N2=None,
                    halstead_length=None,
                    halstead_vocabulary=None,
                    halstead_volume=None,
                    halstead_difficulty=None,
                    halstead_effort=None,
                    halstead_bugs=None,
                    halstead_time=None,
                    nargs=None,
                    nexits=None,
                )
                continue

            halstead = self._extract_halstead_metrics(matched)
            method_metrics[method.method_id] = MethodMetricRecord(
                method_id=method.method_id,
                cc=self._extract_metric_from_space(matched, {"cc", "cyclomatic"}),
                halstead=halstead.get("halstead"),
                halstead_n1=halstead.get("n1"),
                halstead_n2=halstead.get("n2"),
                halstead_N1=halstead.get("N1"),
                halstead_N2=halstead.get("N2"),
                halstead_length=halstead.get("length"),
                halstead_vocabulary=halstead.get("vocabulary"),
                halstead_volume=halstead.get("volume"),
                halstead_difficulty=halstead.get("difficulty"),
                halstead_effort=halstead.get("effort"),
                halstead_bugs=halstead.get("bugs"),
                halstead_time=halstead.get("time"),
                nargs=self._extract_metric_from_space(matched, {"nargs", "arguments"}),
                nexits=self._extract_metric_from_space(matched, {"nexits", "exits"}),
            )
        return file_metrics, method_metrics

    def extract_all_for_modified_file(
        self,
        *,
        logical_file_path: str,
        source_code: str,
        affected_methods_in_file: Iterable[MethodInfo],
    ) -> tuple[FileMetricRecord, Dict[str, MethodMetricRecord]]:
        """Single RCA run per file for file/method metrics."""
        method_list = list(affected_methods_in_file)
        suffix = Path(logical_file_path).suffix or ".txt"
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=suffix, delete=True) as temp_file:
            temp_file.write(source_code)
            temp_file.flush()
            parsed = self._run_and_parse(Path(temp_file.name))
        self._dump_debug_json(logical_file_path, parsed)

        file_metrics = FileMetricRecord(
            file_path=logical_file_path,
            cloc=self._extract_file_metric(parsed, {"cloc", "comment_lines"}),
            mi=self._extract_file_metric(parsed, {"mi", "maintainability_index"}),
            nom=self._extract_file_metric(parsed, {"nom", "functions", "methods_count"}),
        )

        method_metrics: Dict[str, MethodMetricRecord] = {}
        method_spaces = self._collect_method_spaces(parsed)
        for method in method_list:
            matched = self._best_match_method_space(method, method_spaces)
            if matched is None:
                method_metrics[method.method_id] = MethodMetricRecord(
                    method_id=method.method_id,
                    cc=None,
                    halstead=None,
                    halstead_n1=None,
                    halstead_n2=None,
                    halstead_N1=None,
                    halstead_N2=None,
                    halstead_length=None,
                    halstead_vocabulary=None,
                    halstead_volume=None,
                    halstead_difficulty=None,
                    halstead_effort=None,
                    halstead_bugs=None,
                    halstead_time=None,
                    nargs=None,
                    nexits=None,
                )
                continue
            halstead = self._extract_halstead_metrics(matched)
            method_metrics[method.method_id] = MethodMetricRecord(
                method_id=method.method_id,
                cc=self._extract_metric_from_space(matched, {"cc", "cyclomatic"}),
                halstead=halstead.get("halstead"),
                halstead_n1=halstead.get("n1"),
                halstead_n2=halstead.get("n2"),
                halstead_N1=halstead.get("N1"),
                halstead_N2=halstead.get("N2"),
                halstead_length=halstead.get("length"),
                halstead_vocabulary=halstead.get("vocabulary"),
                halstead_volume=halstead.get("volume"),
                halstead_difficulty=halstead.get("difficulty"),
                halstead_effort=halstead.get("effort"),
                halstead_bugs=halstead.get("bugs"),
                halstead_time=halstead.get("time"),
                nargs=self._extract_metric_from_space(matched, {"nargs", "arguments"}),
                nexits=self._extract_metric_from_space(matched, {"nexits", "exits"}),
            )
        return file_metrics, method_metrics

    def _extract_halstead_metrics(self, space: Dict[str, Any]) -> Dict[str, float | None]:
        metrics = self._find_metrics_dict(space) or {}
        result: Dict[str, float | None] = {
            "halstead": self._find_numeric_by_alias(metrics, {"halstead"}),
            "n1": self._find_numeric_by_alias(metrics, {"n1", "n_1", "distinct_operators"}),
            "n2": self._find_numeric_by_alias(metrics, {"n2", "n_2", "distinct_operands"}),
            "N1": self._find_numeric_by_alias(metrics, {"N1", "big_n1", "total_operators"}),
            "N2": self._find_numeric_by_alias(metrics, {"N2", "big_n2", "total_operands"}),
            "length": self._find_numeric_by_alias(metrics, {"length", "program_length"}),
            "vocabulary": self._find_numeric_by_alias(metrics, {"vocabulary", "program_vocabulary"}),
            "volume": self._find_numeric_by_alias(metrics, {"volume"}),
            "difficulty": self._find_numeric_by_alias(metrics, {"difficulty"}),
            "effort": self._find_numeric_by_alias(metrics, {"effort"}),
            "bugs": self._find_numeric_by_alias(metrics, {"bugs", "delivered_bugs"}),
            "time": self._find_numeric_by_alias(metrics, {"time", "time_required"}),
        }
        return result

    def _run_and_parse(self, file_path: Path) -> Dict[str, Any]:
        # rust-code-analysis-cli metrics mode:
        # -m (metrics), -p (path), -O json (output format), --pr (pretty print to stdout)
        cmd = [self.command, "-m", "-p", str(file_path), "-O", "json", "--pr"]
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except Exception as exc:
            self.logger.warning("rust-code-analysis failed for %s: %s", str(file_path), exc)
            return {}

        stdout = result.stdout.strip()
        if not stdout:
            stderr = (result.stderr or "").strip()
            if stderr:
                self.logger.warning("rust-code-analysis empty stdout for %s, stderr=%s", str(file_path), stderr)
            return {}
        try:
            return json.loads(stdout)
        except json.JSONDecodeError:
            stderr = (result.stderr or "").strip()
            self.logger.warning(
                "rust-code-analysis produced non-JSON output for %s. stderr=%s",
                str(file_path),
                stderr,
            )
            return {}

    def _collect_method_spaces(self, obj: Any) -> List[Dict[str, Any]]:
        spaces: List[Dict[str, Any]] = []
        self._walk_collect_spaces(obj, spaces)
        return spaces

    def _walk_collect_spaces(self, obj: Any, spaces: List[Dict[str, Any]]) -> None:
        if isinstance(obj, dict):
            kind = str(obj.get("kind", "")).lower()
            if any(token in kind for token in ("function", "method")):
                spaces.append(obj)
            for value in obj.values():
                self._walk_collect_spaces(value, spaces)
        elif isinstance(obj, list):
            for item in obj:
                self._walk_collect_spaces(item, spaces)

    def _best_match_method_space(self, method: MethodInfo, spaces: List[Dict[str, Any]]) -> Dict[str, Any] | None:
        candidates: List[tuple[int, Dict[str, Any]]] = []
        for space in spaces:
            start_line = self._extract_line(space, {"start_line", "line_start", "start"})
            end_line = self._extract_line(space, {"end_line", "line_end", "end"})
            if start_line is None or end_line is None:
                continue
            if max(start_line, method.start_line) <= min(end_line, method.end_line):
                dist = abs(start_line - method.start_line)
                candidates.append((dist, space))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def _extract_metric_from_space(self, space: Dict[str, Any], aliases: set[str]) -> float | None:
        metrics = self._find_metrics_dict(space)
        if metrics:
            found = self._find_numeric_by_alias(metrics, aliases)
            if found is not None:
                return found
        return self._find_numeric_by_alias(space, aliases)

    def _extract_file_metric(self, root: Dict[str, Any], aliases: set[str]) -> float | None:
        # File-level extraction should prefer file/root metrics, not first nested method metrics.
        metrics = self._find_file_metrics_dict(root, aliases)
        if metrics:
            exact = self._find_numeric_by_exact_key(metrics, aliases)
            if exact is not None:
                return exact
            found = self._find_numeric_by_alias(metrics, aliases)
            if found is not None:
                return found

        # Fallback to whole-tree exact/alias search.
        exact = self._find_numeric_by_exact_key(root, aliases)
        if exact is not None:
            return exact
        return self._find_numeric_by_alias(root, aliases)

    def _find_metrics_dict(self, obj: Any) -> Dict[str, Any] | None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if str(key).lower() == "metrics" and isinstance(value, dict):
                    return value
            for value in obj.values():
                nested = self._find_metrics_dict(value)
                if nested is not None:
                    return nested
        elif isinstance(obj, list):
            for item in obj:
                nested = self._find_metrics_dict(item)
                if nested is not None:
                    return nested
        return None

    def _find_file_metrics_dict(self, root: Any, aliases: set[str]) -> Dict[str, Any] | None:
        """Pick the most likely file-level metrics dict (avoid method-level first-hit bias)."""
        candidates: List[tuple[int, Dict[str, Any]]] = []
        self._collect_metrics_candidates(root, depth=0, candidates=candidates)
        if not candidates:
            return None

        aliases_norm = {alias.lower() for alias in aliases}

        def score(item: tuple[int, Dict[str, Any]]) -> int:
            depth, node = item
            kind = str(node.get("kind", "")).lower()
            metrics = node.get("metrics", {})
            metric_keys = {str(key).lower() for key in metrics.keys()} if isinstance(metrics, dict) else set()

            file_like_tokens = {"unit", "file", "module", "translation_unit", "source_file", "program"}
            method_like_tokens = {"function", "method", "lambda", "closure"}

            s = 0
            # Prefer shallower nodes (typically file/root metrics).
            s += max(0, 30 - depth)
            if any(token in kind for token in file_like_tokens):
                s += 80
            if any(token in kind for token in method_like_tokens):
                s -= 120
            # Prefer metrics dicts that explicitly contain requested aliases.
            if metric_keys & aliases_norm:
                s += 60
            return s

        candidates.sort(key=score, reverse=True)
        best = candidates[0][1]
        metrics = best.get("metrics", {})
        return metrics if isinstance(metrics, dict) else None

    def _collect_metrics_candidates(self, obj: Any, depth: int, candidates: List[tuple[int, Dict[str, Any]]]) -> None:
        if isinstance(obj, dict):
            metrics = obj.get("metrics")
            if isinstance(metrics, dict):
                candidates.append((depth, obj))
            for value in obj.values():
                self._collect_metrics_candidates(value, depth + 1, candidates)
        elif isinstance(obj, list):
            for item in obj:
                self._collect_metrics_candidates(item, depth + 1, candidates)

    def _find_numeric_by_alias(self, obj: Any, aliases: set[str]) -> float | None:
        aliases_norm = {alias.lower() for alias in aliases}
        if isinstance(obj, dict):
            for key, value in obj.items():
                key_norm = str(key).lower()
                if any(alias in key_norm for alias in aliases_norm):
                    numeric = self._extract_number(value)
                    if numeric is not None:
                        return numeric
                nested = self._find_numeric_by_alias(value, aliases_norm)
                if nested is not None:
                    return nested
        elif isinstance(obj, list):
            for item in obj:
                nested = self._find_numeric_by_alias(item, aliases_norm)
                if nested is not None:
                    return nested
        return None

    def _find_numeric_by_exact_key(self, obj: Any, aliases: set[str]) -> float | None:
        aliases_norm = {alias.lower() for alias in aliases}
        if isinstance(obj, dict):
            for key, value in obj.items():
                key_norm = str(key).lower()
                if key_norm in aliases_norm:
                    numeric = self._extract_number(value)
                    if numeric is not None:
                        return numeric
                nested = self._find_numeric_by_exact_key(value, aliases_norm)
                if nested is not None:
                    return nested
        elif isinstance(obj, list):
            for item in obj:
                nested = self._find_numeric_by_exact_key(item, aliases_norm)
                if nested is not None:
                    return nested
        return None

    def _extract_number(self, value: Any) -> float | None:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, dict):
            for key in ("sum", "total", "value", "avg", "mean"):
                if key in value and isinstance(value[key], (int, float)):
                    return float(value[key])
            for nested in value.values():
                number = self._extract_number(nested)
                if number is not None:
                    return number
        if isinstance(value, list):
            for item in value:
                number = self._extract_number(item)
                if number is not None:
                    return number
        return None

    def _extract_line(self, obj: Dict[str, Any], aliases: set[str]) -> int | None:
        for key, value in obj.items():
            key_norm = str(key).lower()
            if any(alias in key_norm for alias in aliases) and isinstance(value, int):
                return value
            if isinstance(value, dict):
                nested = self._extract_line(value, aliases)
                if nested is not None:
                    return nested
        return None

    def _dump_debug_json(self, logical_file_path: str, parsed: Dict[str, Any]) -> None:
        if self.debug_dump_dir is None:
            return
        stable = hashlib.sha1(logical_file_path.encode("utf-8")).hexdigest()[:12]
        name = Path(logical_file_path).name.replace("/", "_").replace("\\", "_")
        out_file = self.debug_dump_dir / f"{name}_{stable}.json"
        try:
            out_file.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:  # pragma: no cover
            self.logger.warning("Failed to dump RCA debug JSON for %s: %s", logical_file_path, exc)

