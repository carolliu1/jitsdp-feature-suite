"""PHP parser using language-tuned generic tree-sitter extraction."""

from __future__ import annotations

from commit_feature_suite.parsers.generic_parser import GenericTreeSitterParser
from commit_feature_suite.parsers.language_profiles import LANGUAGE_PROFILES


class PhpParser(GenericTreeSitterParser):
    """PHP-specific parser wrapper."""

    def __init__(self) -> None:
        super().__init__(
            profile=LANGUAGE_PROFILES["php"],
            import_patterns=(r"^\s*use\s+([a-zA-Z0-9_\\]+)\s*;",),
            receiver_type_patterns=(r"\$([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*new\s+([A-Z][A-Za-z0-9_\\]+)",),
        )

