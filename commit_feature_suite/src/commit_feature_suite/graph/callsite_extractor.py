"""Callsite extraction based on tree-sitter language parsers."""

from __future__ import annotations

import hashlib
from typing import Dict, List, Tuple

from commit_feature_suite.models import CallSite
from commit_feature_suite.parsers import create_language_parser, has_tree_sitter_parser


class ParserRegistry:
    """Cache language parser instances."""

    def __init__(self) -> None:
        self._parsers: Dict[str, object] = {}

    def get(self, language: str):
        """Return a cached parser instance for a language."""
        if language not in self._parsers:
            self._parsers[language] = create_language_parser(language)
        return self._parsers[language]


class TreeSitterCallsiteExtractor:
    """Extract callsites and metadata from source text."""

    def __init__(self, logger) -> None:
        self.logger = logger
        self.registry = ParserRegistry()
        self._tree_cache: Dict[Tuple[str, str], Tuple[str, object]] = {}

    def supported(self, language: str) -> bool:
        """Whether this language has tree-sitter extraction configured."""
        return has_tree_sitter_parser(language)

    def extract(
        self,
        *,
        language: str,
        relative_path: str,
        source_code: str,
    ) -> Tuple[List[CallSite], Dict[str, object]]:
        """Extract callsites and metadata from one source file."""
        try:
            parser = self.registry.get(language)
        except Exception as exc:  # pragma: no cover
            self.logger.warning("Failed to initialize parser for %s (%s): %s", relative_path, language, exc)
            return [], {}
        try:
            parsed = parser.parse_file(source_code, relative_path)
        except Exception as exc:  # pragma: no cover
            self.logger.warning("Failed to parse callsites %s (%s): %s", relative_path, language, exc)
            return [], {}
        metadata = dict(parsed.metadata)
        root = metadata.get("_tree_root")
        if root is not None:
            self._tree_cache[(language, relative_path)] = (self._source_digest(source_code), root)
        return list(parsed.callsites), metadata

    def get_cached_tree_root(self, *, language: str, relative_path: str, source_code: str):
        """Return cached tree root if source content matches, else None."""
        item = self._tree_cache.get((language, relative_path))
        if item is None:
            return None
        cached_digest, root = item
        if cached_digest != self._source_digest(source_code):
            return None
        return root

    @staticmethod
    def _source_digest(source_code: str) -> str:
        return hashlib.sha1(source_code.encode("utf-8")).hexdigest()

