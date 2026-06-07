"""Ruby parser using language-tuned generic tree-sitter extraction."""

from __future__ import annotations

from commit_feature_suite.parsers.generic_parser import GenericTreeSitterParser
from commit_feature_suite.parsers.language_profiles import LANGUAGE_PROFILES


class RubyParser(GenericTreeSitterParser):
    """Ruby-specific parser wrapper."""

    def __init__(self) -> None:
        super().__init__(
            profile=LANGUAGE_PROFILES["ruby"],
            import_patterns=(r"^\s*require(_relative)?\s+['\"]([^'\"]+)['\"]",),
            receiver_type_patterns=(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([A-Z][A-Za-z0-9_:]*)\.new",),
        )

