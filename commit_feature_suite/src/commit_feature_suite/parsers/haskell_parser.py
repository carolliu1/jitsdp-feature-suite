"""Haskell parser using language-tuned generic tree-sitter extraction."""

from __future__ import annotations

from commit_feature_suite.parsers.generic_parser import GenericTreeSitterParser
from commit_feature_suite.parsers.language_profiles import LANGUAGE_PROFILES


class HaskellParser(GenericTreeSitterParser):
    """Haskell-specific parser wrapper."""

    def __init__(self) -> None:
        super().__init__(
            profile=LANGUAGE_PROFILES["haskell"],
            import_patterns=(r"^\s*import\s+(?:qualified\s+)?([A-Z][A-Za-z0-9_.]*)",),
        )

