"""Main analyzer orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from commit_feature_suite.affected.affected_methods import collect_affected_methods
from commit_feature_suite.config import (
    DEFAULT_LANGUAGES,
    LANGUAGE_FILE_EXTENSIONS,
    RuntimeConfig,
)
from commit_feature_suite.gitops.commits import is_merge_commit, iter_commits
from commit_feature_suite.gitops.snapshots import create_snapshot_at_commit, resolve_analysis_repo_path
from commit_feature_suite.features.commit_features import CommitFeatureBuilder
from commit_feature_suite.features.function_features import FunctionFeatureBuilder
from commit_feature_suite.features.rca_features import RCAMetricRowBuilder
from commit_feature_suite.graph.builder import CallGraphBuilder
from commit_feature_suite.graph.result import CallGraphBuildResult
from commit_feature_suite.metrics.rust_code_analysis import (
    FileMetricRecord,
    MethodMetricRecord,
    RustCodeAnalysisExtractor,
)
from commit_feature_suite.metrics.coupling import MethodCouplingCollector
from commit_feature_suite.metrics.global_var_counter import GlobalVariableCounter
from commit_feature_suite.output.writers import FeatureOutputWriter
from commit_feature_suite.results import AnalysisResult, CommitAnalysisResult
from commit_feature_suite.utils import detect_language_from_path, setup_logging


@dataclass
class _SyntheticModifiedFile:
    old_path: str | None
    new_path: str | None
    diff: str
    diff_parsed: Dict[str, List[tuple[int, str]]]
    changed_methods: List[Any]
    source_code_before: str | None
    source_code: str | None
    language_supported: str | None = None
    change_type: Any | None = None


class CommitFeatureSuiteAnalyzer:
    """Main orchestration class."""
    METHOD_AGG_METRICS = [
        "method_in_coupling",
        "method_out_coupling",
        "method_cc",
        "method_halstead",
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
        "method_nargs",
        "method_nexits",
        "method_global_var_count",
    ]
    FILE_AGG_METRICS = ["file_cloc", "file_mi", "file_nom", "file_class"]

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.logger = setup_logging(config.log_level)
        self.repo_path = resolve_analysis_repo_path(
            config.repo_path,
            config.repo_url,
            config.local_repo_path,
            self.logger,
        )
        if config.languages:
            self.languages = list(config.languages)
        else:
            self.languages = self._detect_languages_in_repo(self.repo_path)
            if not self.languages:
                self.languages = list(DEFAULT_LANGUAGES)
            self.logger.info("Auto-detected languages: %s", ",".join(self.languages))

        self.call_graph_builder = CallGraphBuilder(self.languages, self.logger)
        self._graph_cache: Dict[str, CallGraphBuildResult] = {}
        self.output_writer = FeatureOutputWriter()
        self.function_feature_builder = FunctionFeatureBuilder()
        self.commit_feature_builder = CommitFeatureBuilder(self.METHOD_AGG_METRICS, self.FILE_AGG_METRICS)
        self.rca_row_builder = RCAMetricRowBuilder()
        self._rca_extractor = None
        self._global_var_counter = GlobalVariableCounter(self.logger)
        self._coupling_collector = MethodCouplingCollector()
        if config.enable_rca_metrics:
            command_exists = shutil.which(config.rca_command) is not None or Path(config.rca_command).expanduser().exists()
            if command_exists:
                self._rca_extractor = RustCodeAnalysisExtractor(
                    config.rca_command,
                    self.logger,
                    debug_dump_dir=config.rca_debug_dump_dir,
                )
            else:
                self.logger.warning("RCA command not found: %s. RCA metrics disabled.", config.rca_command)

    def run(self) -> None:
        """Execute analysis and write results to CSV."""
        all_results = AnalysisResult()

        commits = list(
            iter_commits(
                self.repo_path,
                max_commits=self.config.max_commits,
                skip_commits=self.config.skip_commits,
            )
        )
        self.logger.info(
            "Commit window: skip_commits=%d, max_commits=%s, selected=%d",
            self.config.skip_commits,
            str(self.config.max_commits),
            len(commits),
        )
        for commit in tqdm(commits, desc="Analyzing commits"):
            result = self._analyze_commit(
                commit=commit,
                builder=self.call_graph_builder,
                graph_cache=self._graph_cache,
            )
            all_results.extend(result)

        self.output_writer.write_feature_tables(
            function_rows=all_results.function_rows,
            file_metric_rows=all_results.file_metric_rows,
            commit_rows=all_results.commit_rows,
            output_prefix=self.config.output_csv,
        )
        self.logger.info(
            "Finished. Processed %d commits and wrote %d function rows.",
            len(commits),
            len(all_results.function_rows),
        )

    def _analyze_commit(
        self,
        *,
        commit,
        builder: CallGraphBuilder,
        graph_cache: Dict[str, CallGraphBuildResult],
    ) -> CommitAnalysisResult:
        """Analyze one commit and return function/file/commit rows."""
        commit_rows: List[Dict[str, Any]] = []
        file_metric_rows: List[Dict[str, Any]] = []
        effective_modified_files = self._effective_modified_files_for_commit(commit)
        snapshot_targets = self._snapshot_targets_for_commit(commit)
        if not snapshot_targets:
            return CommitAnalysisResult(
                function_rows=commit_rows,
                file_metric_rows=file_metric_rows,
                commit_rows=[],
            )
        current_new_file_count, current_new_file_ratio = self.commit_feature_builder.current_new_file_stats(effective_modified_files)

        commit_current_in_values: List[int] = []
        commit_current_out_values: List[int] = []
        (
            scoped_method_metrics,
            scoped_file_metrics,
            scoped_modified_method_ids,
            class_touched_count_by_scope,
        ) = self.commit_feature_builder.init_scope_buffers()
        for snapshot_scope, _, snapshot_hash in snapshot_targets:
            graph_result = self._get_or_build_snapshot_graph(snapshot_hash, builder, graph_cache)
            if graph_result is None:
                self.logger.warning("Skip commit %s snapshot %s due to unavailable snapshot", commit.hash, snapshot_hash)
                continue

            source_modified_files, non_coupling_modified_files = self._split_modified_files_by_language(
                effective_modified_files,
                snapshot_scope=snapshot_scope,
            )

            affected_methods = collect_affected_methods(
                modified_files=source_modified_files,
                methods_by_file=graph_result.methods_by_file,
                snapshot_scope=snapshot_scope,
                strict_diff_mapping=True,
                logger=self.logger,
            )
            method_coupling_map = self._coupling_collector.build_method_coupling_map(
                graph_result=graph_result,
                affected_methods=affected_methods,
            )
            scope_in_values, scope_out_values = self._coupling_collector.degree_values(
                method_coupling_map=method_coupling_map,
            )
            commit_current_in_values.extend(scope_in_values)
            commit_current_out_values.extend(scope_out_values)

            touched_classes_scope = self._collect_touched_classes(affected_methods)
            file_metric_map, method_metric_map = self._collect_rca_metrics_for_snapshot(
                modified_files=source_modified_files + non_coupling_modified_files,
                affected_methods=affected_methods,
                snapshot_scope=snapshot_scope,
                methods_by_file=graph_result.methods_by_file,
            )
            global_var_count_map = self._collect_global_var_counts_for_snapshot(
                modified_files=source_modified_files + non_coupling_modified_files,
                affected_methods=affected_methods,
                snapshot_scope=snapshot_scope,
            )
            self.commit_feature_builder.accumulate_scope_metrics(
                scope=snapshot_scope,
                affected_methods=affected_methods,
                method_metric_map=method_metric_map,
                method_coupling_map=method_coupling_map,
                global_var_count_map=global_var_count_map,
                file_metric_map=file_metric_map,
                scoped_method_metrics=scoped_method_metrics,
                scoped_file_metrics=scoped_file_metrics,
            )
            scoped_modified_method_ids[snapshot_scope].update(item.method.method_id for item in affected_methods)
            method_info_map = {item.method.method_id: item.method for item in affected_methods}
            file_metric_rows.extend(
                self.rca_row_builder.build_metric_rows(
                    commit=commit,
                    snapshot_scope=snapshot_scope,
                    snapshot_hash=snapshot_hash,
                    method_metric_map=method_metric_map,
                    file_metric_map=file_metric_map,
                    method_info_map=method_info_map,
                )
            )
            for affected in affected_methods:
                commit_rows.append(
                    self.function_feature_builder.build_output_row(
                        commit=commit,
                        graph_result=graph_result,
                        affected_method=affected,
                        snapshot_scope=snapshot_scope,
                        snapshot_hash=snapshot_hash,
                        method_metric_map=method_metric_map,
                        file_metric_map=file_metric_map,
                        global_var_count_map=global_var_count_map,
                        method_coupling_map=method_coupling_map,
                    )
                )

            for modified_file in non_coupling_modified_files:
                commit_rows.append(
                    self.function_feature_builder.build_non_method_coupling_row(
                        commit=commit,
                        graph_result=graph_result,
                        modified_file=modified_file,
                        snapshot_scope=snapshot_scope,
                        snapshot_hash=snapshot_hash,
                        file_metric_map=file_metric_map,
                    )
                )

            class_touched_count_by_scope[snapshot_scope] = len(touched_classes_scope)

        commit_scope_features = self.commit_feature_builder.build_commit_feature_row(
            commit=commit,
            scoped_method_metrics=scoped_method_metrics,
            scoped_file_metrics=scoped_file_metrics,
            scoped_modified_method_ids=scoped_modified_method_ids,
            current_new_file_count=current_new_file_count,
            current_new_file_ratio=current_new_file_ratio,
            class_touched_count_by_scope=class_touched_count_by_scope,
        )
        return CommitAnalysisResult(
            function_rows=commit_rows,
            file_metric_rows=file_metric_rows,
            commit_rows=[commit_scope_features],
        )

    @staticmethod
    def _collect_touched_classes(affected_methods) -> set[tuple[str, str]]:
        touched: set[tuple[str, str]] = set()
        for item in affected_methods:
            method = item.method
            class_name = (method.class_name or "").strip()
            if class_name:
                touched.add((method.file_path, class_name))
        return touched

    def _effective_modified_files_for_commit(self, commit) -> List[Any]:
        """Return modified files; merge commits are forced to first-parent effective diff."""
        modified_files = list(getattr(commit, "modified_files", []) or [])
        if not is_merge_commit(commit):
            return modified_files
        parents = list(getattr(commit, "parents", []) or [])
        if not parents:
            return modified_files
        first_parent = parents[0]
        self.logger.info(
            "Merge commit %s: force reconstruct modified files from first parent %s",
            commit.hash,
            first_parent,
        )
        rebuilt_files = self._build_synthetic_modified_files(first_parent, commit.hash)
        self.logger.debug(
            "Merge commit %s: rebuilt modified_files count=%d (first_parent=%s)",
            commit.hash,
            len(rebuilt_files),
            first_parent,
        )
        return rebuilt_files

    def _build_synthetic_modified_files(self, parent_hash: str, commit_hash: str) -> List[_SyntheticModifiedFile]:
        files: List[_SyntheticModifiedFile] = []
        name_status = self._git_capture(["diff", "--name-status", "-M", parent_hash, commit_hash])
        if not name_status:
            return files
        for line in name_status.splitlines():
            parts = line.strip().split("\t")
            if not parts:
                continue
            status = parts[0]
            old_path: str | None = None
            new_path: str | None = None
            if status.startswith("R") and len(parts) >= 3:
                old_path, new_path = parts[1], parts[2]
            elif status.startswith("A") and len(parts) >= 2:
                old_path, new_path = None, parts[1]
            elif status.startswith("D") and len(parts) >= 2:
                old_path, new_path = parts[1], None
            elif len(parts) >= 2:
                old_path, new_path = parts[1], parts[1]
            else:
                continue

            target_path = new_path or old_path
            if not target_path:
                continue
            diff_text = self._git_capture(["diff", "-U0", parent_hash, commit_hash, "--", target_path]) or ""
            diff_parsed = self._parse_unified_diff_lines(diff_text)
            source_before = self._git_show_file(parent_hash, old_path) if old_path else None
            source_after = self._git_show_file(commit_hash, new_path) if new_path else None
            files.append(
                _SyntheticModifiedFile(
                    old_path=old_path,
                    new_path=new_path,
                    diff=diff_text,
                    diff_parsed=diff_parsed,
                    changed_methods=[],
                    source_code_before=source_before,
                    source_code=source_after,
                )
            )
        return files

    def _git_capture(self, args: List[str]) -> str:
        try:
            proc = subprocess.run(
                ["git", "-C", str(self.repo_path), *args],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.CalledProcessError as exc:
            self.logger.warning("git command failed: %s (%s)", " ".join(args), exc)
            return ""
        return proc.stdout

    def _git_show_file(self, commit_hash: str, path: str | None) -> str | None:
        if not path:
            return None
        try:
            proc = subprocess.run(
                ["git", "-C", str(self.repo_path), "show", f"{commit_hash}:{path}"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
            )
        except subprocess.CalledProcessError:
            return None
        return proc.stdout

    @staticmethod
    def _parse_unified_diff_lines(diff_text: str) -> Dict[str, List[tuple[int, str]]]:
        added: List[tuple[int, str]] = []
        deleted: List[tuple[int, str]] = []
        old_line = 0
        new_line = 0
        in_hunk = False
        for raw in diff_text.splitlines():
            if raw.startswith("@@"):
                in_hunk = True
                # @@ -a,b +c,d @@
                parts = raw.split()
                if len(parts) >= 3:
                    old_info = parts[1]  # -a,b
                    new_info = parts[2]  # +c,d
                    try:
                        old_line = int(old_info[1:].split(",", 1)[0])
                        new_line = int(new_info[1:].split(",", 1)[0])
                    except ValueError:
                        old_line = 0
                        new_line = 0
                continue
            if not in_hunk:
                continue
            if raw.startswith("+") and not raw.startswith("+++"):
                added.append((new_line, raw[1:]))
                new_line += 1
            elif raw.startswith("-") and not raw.startswith("---"):
                deleted.append((old_line, raw[1:]))
                old_line += 1
            else:
                old_line += 1
                new_line += 1
        return {"added": added, "deleted": deleted}

    def _snapshot_targets_for_commit(self, commit) -> List[Tuple[str, int, str]]:
        """Build snapshot targets for one commit.

        Current feature pipeline only emits current-snapshot features.
        """
        return [("current", 0, commit.hash)]

    def _get_or_build_snapshot_graph(
        self,
        snapshot_hash: str,
        builder: CallGraphBuilder,
        graph_cache: Dict[str, CallGraphBuildResult],
    ) -> CallGraphBuildResult | None:
        cached = graph_cache.get(snapshot_hash)
        if cached is not None:
            return cached

        snapshot_context = None
        try:
            snapshot_context = create_snapshot_at_commit(self.repo_path, snapshot_hash)
            graph_result = builder.build_from_snapshot(snapshot_context.snapshot_path)
            self._remember_graph(snapshot_hash, graph_result, graph_cache)
            return graph_result
        except Exception as exc:  # pragma: no cover
            self.logger.error("Failed to build snapshot graph for %s: %s", snapshot_hash, exc)
            return None
        finally:
            if snapshot_context is not None:
                snapshot_context.cleanup()

    def _remember_graph(
        self,
        snapshot_hash: str,
        graph_result: CallGraphBuildResult,
        graph_cache: Dict[str, CallGraphBuildResult],
    ) -> None:
        graph_cache[snapshot_hash] = graph_result
        if len(graph_cache) > 6:
            oldest_key = next(iter(graph_cache))
            del graph_cache[oldest_key]

    def _collect_global_var_counts_for_snapshot(
        self,
        *,
        modified_files: List[Any],
        affected_methods,
        snapshot_scope: str,
    ) -> Dict[str, int]:
        if not affected_methods:
            return {}
        affected_by_file: Dict[str, List[Any]] = {}
        for affected in affected_methods:
            affected_by_file.setdefault(affected.method.file_path, []).append(affected.method)

        result: Dict[str, int] = {}
        seen_files: set[str] = set()
        for modified_file in modified_files:
            old_path = getattr(modified_file, "old_path", None)
            new_path = getattr(modified_file, "new_path", None)
            logical_path = (old_path or new_path) if snapshot_scope == "parent" else (new_path or old_path)
            if not logical_path or logical_path in seen_files:
                continue
            seen_files.add(logical_path)
            methods_in_file = affected_by_file.get(logical_path, [])
            if not methods_in_file:
                continue

            language = self._language_of_modified_file(modified_file, snapshot_scope=snapshot_scope)
            if language is None:
                continue
            source_attr = "source_code_before" if snapshot_scope == "parent" else "source_code"
            try:
                source_code = getattr(modified_file, source_attr, None)
            except Exception:
                continue
            if not source_code:
                continue
            cached_root = self.call_graph_builder.callsite_extractor.get_cached_tree_root(
                language=language,
                relative_path=logical_path,
                source_code=source_code,
            )
            if cached_root is not None:
                file_counts = self._global_var_counter.count_for_methods_with_root(
                    language=language,
                    source_code=source_code,
                    methods=methods_in_file,
                    root=cached_root,
                )
            else:
                file_counts = self._global_var_counter.count_for_methods(
                    language=language,
                    source_code=source_code,
                    methods=methods_in_file,
                )
            result.update(file_counts)
        return result

    def _collect_rca_metrics_for_snapshot(
        self,
        *,
        modified_files: List[Any],
        affected_methods,
        snapshot_scope: str,
        methods_by_file: Dict[str, List[Any]],
    ) -> tuple[Dict[str, FileMetricRecord], Dict[str, MethodMetricRecord]]:
        if self._rca_extractor is None:
            return {}, {}

        affected_by_file: Dict[str, List[Any]] = {}
        for affected in affected_methods:
            affected_by_file.setdefault(affected.method.file_path, []).append(affected.method)

        class_count_cache: Dict[str, int] = {}

        file_metric_map: Dict[str, FileMetricRecord] = {}
        method_metric_map: Dict[str, MethodMetricRecord] = {}
        seen_paths: set[str] = set()
        for modified_file in modified_files:
            old_path = getattr(modified_file, "old_path", None)
            new_path = getattr(modified_file, "new_path", None)
            logical_path = (old_path or new_path) if snapshot_scope == "parent" else (new_path or old_path)
            if not logical_path or logical_path in seen_paths:
                continue
            seen_paths.add(logical_path)
            language = self._language_of_modified_file(modified_file, snapshot_scope=snapshot_scope)
            if language is None or language not in self.languages:
                # RCA only analyzes languages selected by --languages (or auto-detected list).
                continue

            source_attr = "source_code_before" if snapshot_scope == "parent" else "source_code"
            try:
                source_code = getattr(modified_file, source_attr, None)
            except Exception as exc:
                self.logger.warning(
                    "Skip unreadable source for RCA: snapshot_scope=%s file=%s attr=%s error=%s",
                    snapshot_scope,
                    logical_path,
                    source_attr,
                    exc,
                )
                continue
            if not source_code:
                continue

            file_metrics, method_metrics = self._rca_extractor.extract_all_for_modified_file(
                logical_file_path=logical_path,
                source_code=source_code,
                affected_methods_in_file=affected_by_file.get(logical_path, []),
            )
            if logical_path not in class_count_cache:
                methods = methods_by_file.get(logical_path, [])
                class_names = {
                    method.class_name
                    for method in methods
                    if getattr(method, "class_name", None) and str(method.class_name).strip()
                }
                class_count_cache[logical_path] = len(class_names)
            file_metrics.file_class = float(class_count_cache.get(logical_path, 0))
            file_metric_map[logical_path] = file_metrics
            method_metric_map.update(method_metrics)
        return file_metric_map, method_metric_map

    def _detect_languages_in_repo(self, repo_path: Path) -> List[str]:
        ext_to_language: Dict[str, str] = {}
        for language, extensions in LANGUAGE_FILE_EXTENSIONS.items():
            for extension in extensions:
                ext_to_language[extension] = language
        detected: set[str] = set()
        for file_path in repo_path.rglob("*"):
            if not file_path.is_file():
                continue
            if ".git" in file_path.parts:
                continue
            language = ext_to_language.get(file_path.suffix.lower())
            if language:
                detected.add(language)
        ordered = [language for language in DEFAULT_LANGUAGES if language in detected]
        ordered.extend([language for language in sorted(detected) if language not in ordered])
        return ordered

    def _split_modified_files_by_language(self, modified_files, snapshot_scope: str) -> tuple[List[Any], List[Any]]:
        """Split modified files into method-coupling and non-coupling groups.

        - recognized language -> coupling_group
        - unknown/unrecognized language -> non_coupling_group
        """
        coupling_group: List[Any] = []
        non_coupling_group: List[Any] = []
        for modified_file in modified_files:
            language = self._language_of_modified_file(modified_file, snapshot_scope=snapshot_scope)
            if language is not None:
                coupling_group.append(modified_file)
            else:
                # Unknown files are kept for non-method coupling placeholder rows.
                non_coupling_group.append(modified_file)
        return coupling_group, non_coupling_group

    def _language_of_modified_file(self, modified_file, snapshot_scope: str) -> str | None:
        old_path = getattr(modified_file, "old_path", None)
        new_path = getattr(modified_file, "new_path", None)
        if snapshot_scope == "parent":
            candidate_path = old_path or new_path
        else:
            candidate_path = new_path or old_path
        if not candidate_path:
            return None
        return detect_language_from_path(Path(candidate_path), self.languages)


