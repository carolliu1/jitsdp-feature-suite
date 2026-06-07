"""TypeScript parser built on top of the JavaScript parser logic."""

from __future__ import annotations

from commit_feature_suite.parsers.javascript_parser import JavaScriptParser


class TypeScriptParser(JavaScriptParser):
    """Reuse JavaScript extraction heuristics for TypeScript."""

    language_name = "typescript"

