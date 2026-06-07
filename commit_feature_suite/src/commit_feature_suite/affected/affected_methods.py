"""Resolve diff-affected methods for commit snapshots.

This module maps modified files + diff evidence to directly affected methods.
It does not perform call-graph propagation (i.e. callers/callees are not
automatically included unless they are directly modified by the commit).
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from commit_feature_suite.models import MethodInfo


@dataclass(frozen=True)
class AffectedMethodRecord:
    """A graph-resolved method associated with a modified file."""

    method: MethodInfo
    old_path: str | None
    new_path: str | None
    language: str | None


_HUNK_OLD_RE = re.compile(r"^@@\s*-(\d+)(?:,(\d+))?\s+\+\d+(?:,\d+)?\s*@@")
_HUNK_NEW_RE = re.compile(r"^@@\s*-\d+(?:,\d+)?\s+\+(\d+)(?:,(\d+))?\s*@@")


def collect_affected_methods(
    *,
    modified_files: Iterable,
    methods_by_file: dict[str, list[MethodInfo]],
    snapshot_scope: str = "current",
    strict_diff_mapping: bool = False,
    logger,
) -> list[AffectedMethodRecord]:
    """Collect methods directly affected by commit diff for one snapshot scope."""
    results: list[AffectedMethodRecord] = []
    seen_method_ids: set[str] = set()

    for modified_file in modified_files:
        old_path = getattr(modified_file, "old_path", None)
        new_path = getattr(modified_file, "new_path", None)
        graph_file_path = old_path if snapshot_scope == "parent" else (new_path or old_path)
        if not graph_file_path:
            continue

        graph_methods = methods_by_file.get(graph_file_path, [])
        if not graph_methods:
            continue

        target_methods = _select_method_candidates(
            modified_file=modified_file,
            snapshot_scope=snapshot_scope,
            graph_methods=graph_methods,
            strict_diff_mapping=strict_diff_mapping,
            logger=logger,
        )

        for target in target_methods:
            if isinstance(target, MethodInfo):
                resolved = target
            else:
                resolved = _match_method(target, graph_methods)
            if resolved is None or resolved.method_id in seen_method_ids:
                continue
            results.append(
                AffectedMethodRecord(
                    method=resolved,
                    old_path=old_path,
                    new_path=new_path,
                    language=getattr(modified_file, "language_supported", None),
                )
            )
            seen_method_ids.add(resolved.method_id)

    logger.debug("Collected %d affected methods for snapshot_scope=%s", len(results), snapshot_scope)
    return results


def _select_method_candidates(
    *,
    modified_file,
    snapshot_scope: str,
    graph_methods: list[MethodInfo],
    strict_diff_mapping: bool,
    logger,
) -> list:
    changed_methods = list(getattr(modified_file, "changed_methods", []) or [])
    old_path = getattr(modified_file, "old_path", None)
    new_path = getattr(modified_file, "new_path", None)

    if _is_added_file(modified_file) and snapshot_scope == "current" and graph_methods:
        return _log_selected_candidates(
            modified_file=modified_file,
            snapshot_scope=snapshot_scope,
            graph_methods=graph_methods,
            changed_methods=changed_methods,
            impacted_lines_count=0,
            strict_hits_count=0,
            selected=list(graph_methods),
            reason="current_added_file_all_methods",
            logger=logger,
        )

    if strict_diff_mapping:
        impacted_lines = _collect_impacted_lines(modified_file, snapshot_scope=snapshot_scope)
        strict_hits = []
        if impacted_lines:
            strict_hits = [method for method in graph_methods if _method_intersects_lines(method, impacted_lines)]

        if strict_hits and changed_methods:
            selected = _dedupe_methods(strict_hits + changed_methods)
            return _log_selected_candidates(
                modified_file=modified_file,
                snapshot_scope=snapshot_scope,
                graph_methods=graph_methods,
                changed_methods=changed_methods,
                impacted_lines_count=len(impacted_lines),
                strict_hits_count=len(strict_hits),
                selected=selected,
                reason="strict_union_changed_methods",
                logger=logger,
            )
        if strict_hits:
            return _log_selected_candidates(
                modified_file=modified_file,
                snapshot_scope=snapshot_scope,
                graph_methods=graph_methods,
                changed_methods=changed_methods,
                impacted_lines_count=len(impacted_lines),
                strict_hits_count=len(strict_hits),
                selected=strict_hits,
                reason="strict_hits_only",
                logger=logger,
            )
        if changed_methods:
            return _log_selected_candidates(
                modified_file=modified_file,
                snapshot_scope=snapshot_scope,
                graph_methods=graph_methods,
                changed_methods=changed_methods,
                impacted_lines_count=len(impacted_lines),
                strict_hits_count=len(strict_hits),
                selected=changed_methods,
                reason="changed_methods_only",
                logger=logger,
            )
        return _log_selected_candidates(
            modified_file=modified_file,
            snapshot_scope=snapshot_scope,
            graph_methods=graph_methods,
            changed_methods=changed_methods,
            impacted_lines_count=len(impacted_lines),
            strict_hits_count=len(strict_hits),
            selected=[],
            reason="empty_no_selected_methods",
            logger=logger,
        )

    if changed_methods:
        return _log_selected_candidates(
            modified_file=modified_file,
            snapshot_scope=snapshot_scope,
            graph_methods=graph_methods,
            changed_methods=changed_methods,
            impacted_lines_count=0,
            strict_hits_count=0,
            selected=changed_methods,
            reason="changed_methods_only_no_strict",
            logger=logger,
        )
    return _log_selected_candidates(
        modified_file=modified_file,
        snapshot_scope=snapshot_scope,
        graph_methods=graph_methods,
        changed_methods=changed_methods,
        impacted_lines_count=0,
        strict_hits_count=0,
        selected=[],
        reason="empty_no_selected_methods",
        logger=logger,
    )


def _is_added_file(modified_file) -> bool:
    old_path = getattr(modified_file, "old_path", None)
    new_path = getattr(modified_file, "new_path", None)
    change_type = getattr(modified_file, "change_type", None)
    change_name = str(getattr(change_type, "name", "")).upper() if change_type is not None else ""
    return change_name == "ADD" or (old_path is None and new_path is not None)


def _dedupe_methods(methods: list) -> list:
    seen = set()
    result = []
    for method in methods:
        key = (
            getattr(method, "file_path", None),
            getattr(method, "name", None) or getattr(method, "method_name", None),
            getattr(method, "start_line", None),
            getattr(method, "end_line", None),
        )
        if all(value is None for value in key):
            key = repr(method)
        if key in seen:
            continue
        seen.add(key)
        result.append(method)
    return result


def _log_selected_candidates(
    *,
    modified_file,
    snapshot_scope: str,
    graph_methods: list[MethodInfo],
    changed_methods: list,
    impacted_lines_count: int,
    strict_hits_count: int,
    selected: list,
    reason: str,
    logger,
) -> list:
    logger.debug(
        "Affected candidates: scope=%s old_path=%s new_path=%s graph_methods=%d changed_methods=%d impacted_lines=%d strict_hits=%d selected=%d reason=%s",
        snapshot_scope,
        getattr(modified_file, "old_path", None),
        getattr(modified_file, "new_path", None),
        len(graph_methods),
        len(changed_methods),
        impacted_lines_count,
        strict_hits_count,
        len(selected),
        reason,
    )
    return selected


def _match_method(target_method, graph_methods: list[MethodInfo]) -> MethodInfo | None:
    target_name = _method_name(target_method)
    if not target_name:
        return None

    start_line = getattr(target_method, "start_line", None)
    end_line = getattr(target_method, "end_line", None)
    long_name = getattr(target_method, "long_name", "") or ""

    same_name = [method for method in graph_methods if method.method_name == target_name]
    if not same_name:
        return None

    if long_name:
        class_hint = _extract_class_name(long_name, target_name)
        class_matches = [method for method in same_name if method.class_name == class_hint]
        if class_matches:
            same_name = class_matches

    if start_line is not None and end_line is not None:
        ranged = [
            method
            for method in same_name
            if _line_ranges_overlap(start_line, end_line, method.start_line, method.end_line)
        ]
        if ranged:
            same_name = ranged

    if start_line is not None:
        same_name = sorted(same_name, key=lambda item: abs(item.start_line - start_line))

    return same_name[0] if same_name else None


def _method_name(method_obj) -> str:
    name = (getattr(method_obj, "name", None) or getattr(method_obj, "method_name", None) or "")
    if "::" in name:
        return name.split("::")[-1]
    if "." in name:
        name = name.split(".")[-1]
    if "(" in name:
        name = name.split("(", 1)[0]
    if "<" in name and ">" in name and "." not in name:
        name = name.split("<", 1)[0]
    return name


def _extract_class_name(long_name: str, method_name: str) -> str:
    cleaned = long_name.strip()
    if not cleaned:
        return ""
    for separator in ("::", "."):
        token = f"{separator}{method_name}"
        if token in cleaned:
            prefix = cleaned.rsplit(token, 1)[0]
            parts = [part for part in prefix.split(separator) if part]
            return parts[-1] if parts else ""
    return ""


def _line_ranges_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return max(a_start, b_start) <= min(a_end, b_end)


def _method_intersects_lines(method: MethodInfo, impacted_lines: set[int]) -> bool:
    if not impacted_lines:
        return False
    for line in impacted_lines:
        if method.start_line <= line <= method.end_line:
            return True
    return False


def _collect_impacted_lines(modified_file, *, snapshot_scope: str) -> set[int]:
    impacted: set[int] = set()
    diff_parsed = getattr(modified_file, "diff_parsed", None) or {}
    if snapshot_scope == "parent":
        deleted = diff_parsed.get("deleted", []) or []
        for entry in deleted:
            if isinstance(entry, (list, tuple)) and entry:
                line_no = entry[0]
                if isinstance(line_no, int):
                    impacted.add(line_no)
    else:
        added = diff_parsed.get("added", []) or []
        for entry in added:
            if isinstance(entry, (list, tuple)) and entry:
                line_no = entry[0]
                if isinstance(line_no, int):
                    impacted.add(line_no)

    diff_text = getattr(modified_file, "diff", "") or ""
    if not diff_text:
        return impacted
    pattern = _HUNK_OLD_RE if snapshot_scope == "parent" else _HUNK_NEW_RE
    for raw_line in diff_text.splitlines():
        match = pattern.match(raw_line.strip())
        if not match:
            continue
        start_line = int(match.group(1))
        line_span = int(match.group(2) or "1")
        for line_no in range(start_line, start_line + line_span):
            impacted.add(line_no)
    return impacted

