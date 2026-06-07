"""Swift parser using language-tuned generic tree-sitter extraction."""

from __future__ import annotations

from commit_feature_suite.parsers.generic_parser import GenericTreeSitterParser
from commit_feature_suite.parsers.language_profiles import LANGUAGE_PROFILES


class SwiftParser(GenericTreeSitterParser):
    """Swift-specific parser wrapper."""

    def __init__(self) -> None:
        super().__init__(
            profile=LANGUAGE_PROFILES["swift"],
            import_patterns=(r"^\s*import\s+([a-zA-Z0-9_.]+)",),
            receiver_type_patterns=(r"\blet\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*([A-Z][A-Za-z0-9_<>]*)",),
        )

