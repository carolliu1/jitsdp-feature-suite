"""Build method-level call graphs from repository snapshots."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import networkx as nx

from commit_feature_suite.graph.callsite_extractor import TreeSitterCallsiteExtractor
from commit_feature_suite.graph.indexes import build_indexes
from commit_feature_suite.graph.method_extractor import LizardMethodExtractor
from commit_feature_suite.graph.resolver import CallGraphResolver
from commit_feature_suite.graph.result import CallGraphBuildResult
from commit_feature_suite.models import CallSite, MethodInfo
from commit_feature_suite.utils import detect_language_from_path, normalize_rel_path


class CallGraphBuilder:
    """Build a higher-precision method call graph from a source snapshot."""

    def __init__(self, languages: Iterable[str], logger) -> None:
        self.languages = list(languages)
        self.logger = logger
        self.method_extractor = LizardMethodExtractor(logger)
        self.callsite_extractor = TreeSitterCallsiteExtractor(logger)
        self.resolver = CallGraphResolver(logger)
        self._language_cache: Dict[str, str | None] = {}

    def build_from_snapshot(self, snapshot_path: Path) -> CallGraphBuildResult:
        """Scan a snapshot and build the method-level call graph."""
        graph = nx.DiGraph()
        methods_by_id: Dict[str, MethodInfo] = {}
        methods_by_file: Dict[str, List[MethodInfo]] = defaultdict(list)
        callsites_by_file: Dict[str, List[CallSite]] = defaultdict(list)
        file_metadata: Dict[str, Dict[str, object]] = {}
        scanned_source_files = 0

        for file_path in snapshot_path.rglob("*"):
            if not file_path.is_file():
                continue
            relative_path = normalize_rel_path(snapshot_path, file_path)
            language = self._detect_language(relative_path)
            if language is None:
                continue
            scanned_source_files += 1

            methods = self.method_extractor.extract(file_path, relative_path, language)
            self.logger.debug(
                "Lizard extracted %d methods in %s (%s)",
                len(methods),
                relative_path,
                language,
            )
            if not methods:
                fallback_methods = self._extract_methods_with_tree_sitter_fallback(
                    file_path=file_path,
                    relative_path=relative_path,
                    language=language,
                )
                if fallback_methods:
                    self.logger.debug(
                        "Lizard found 0 methods, tree-sitter fallback found %d methods in %s",
                        len(fallback_methods),
                        relative_path,
                    )
                    methods = fallback_methods

            for method in methods:
                methods_by_id[method.method_id] = method
                methods_by_file[method.file_path].append(method)
                graph.add_node(method.method_id, **method.to_node_attributes())

            if not methods:
                continue

            if not self.callsite_extractor.supported(language):
                continue
            try:
                source_code = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                self.logger.debug("Skip non-text source file for call extraction: %s", relative_path)
                continue
            except OSError as exc:
                self.logger.warning("Failed to read file %s: %s", relative_path, exc)
                continue

            callsites, metadata = self.callsite_extractor.extract(
                language=language,
                relative_path=relative_path,
                source_code=source_code,
            )
            file_metadata[relative_path] = metadata
            callsites_by_file[relative_path].extend(callsites)

        indexes = build_indexes(methods_by_id)
        self.resolver.add_edges(
            graph=graph,
            methods_by_file=dict(methods_by_file),
            methods_by_id=methods_by_id,
            callsites_by_file=dict(callsites_by_file),
            file_metadata=file_metadata,
            indexes=indexes,
        )
        return CallGraphBuildResult(
            graph=graph,
            methods_by_id=methods_by_id,
            methods_by_file=dict(methods_by_file),
            node_count=graph.number_of_nodes(),
            file_count=scanned_source_files,
        )

    def _detect_language(self, relative_path: str) -> str | None:
        """Cached language detection based on file extension."""
        if relative_path not in self._language_cache:
            detected = detect_language_from_path(Path(relative_path), self.languages)
            if detected is None and "other" in self.languages and self.resolver.looks_like_source_file(relative_path):
                detected = "other"
            self._language_cache[relative_path] = detected
        return self._language_cache[relative_path]

    def _extract_methods_with_tree_sitter_fallback(
        self,
        *,
        file_path: Path,
        relative_path: str,
        language: str,
    ) -> List[MethodInfo]:
        try:
            source_code = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            self.logger.debug(
                "Skip tree-sitter method fallback for non-utf8 file: %s",
                relative_path,
            )
            return []
        except OSError as exc:
            self.logger.warning(
                "Failed to read file for tree-sitter method fallback %s: %s",
                relative_path,
                exc,
            )
            return []

        try:
            if not self.callsite_extractor.supported(language):
                return []

            parser = self.callsite_extractor.registry.get(language)
            if parser is None:
                return []
            parsed = parser.parse_file(source_code, relative_path)
            return list(getattr(parsed, "methods", []) or [])
        except Exception as exc:
            self.logger.warning(
                "tree-sitter method fallback failed for %s language=%s: %s",
                relative_path,
                language,
                exc,
            )
            return []

