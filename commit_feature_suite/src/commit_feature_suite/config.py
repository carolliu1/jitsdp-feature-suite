"""Project-level configuration constants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

DEFAULT_LANGUAGES: List[str] = [
    "bash",
    "shell",
    "c",
    "coffeescript",
    "cpp",
    "csharp",
    "cmake",
    "css",
    "dart",
    "go",
    "groovy",
    "haskell",
    "java",
    "javascript",
    "julia",
    "kotlin",
    "lisp",
    "lua",
    "ocaml",
    "objectivec",
    "perl",
    "php",
    "python",
    "r",
    "ruby",
    "rust",
    "swift",
    "tcl",
    "typescript",
    "tsx",
    "yaml",
    "xml",
    "zig",
]

LANGUAGE_FILE_EXTENSIONS: Dict[str, Set[str]] = {
    "bash": {".sh", ".bash", ".zsh", ".ksh"},
    "shell": {".sh", ".bash", ".zsh", ".ksh"},
    "c": {".c", ".h"},
    "coffeescript": {".coffee", ".litcoffee"},
    "cpp": {".cc", ".cpp", ".cxx", ".hpp", ".hh", ".hxx"},
    "csharp": {".cs"},
    "cmake": {".cmake"},
    "css": {".css"},
    "dart": {".dart"},
    "go": {".go"},
    "groovy": {".groovy", ".gradle"},
    "haskell": {".hs", ".lhs"},
    "java": {".java"},
    "javascript": {".js", ".jsx", ".mjs", ".cjs"},
    "julia": {".jl"},
    "kotlin": {".kt", ".kts"},
    "lisp": {".lisp", ".lsp", ".cl"},
    "lua": {".lua"},
    "ocaml": {".ml", ".mli"},
    "objectivec": {".m", ".mm"},
    "perl": {".pl", ".pm", ".t"},
    "php": {".php"},
    "python": {".py"},
    "r": {".r"},
    "ruby": {".rb"},
    "rust": {".rs"},
    "swift": {".swift"},
    "tcl": {".tcl"},
    "typescript": {".ts"},
    "tsx": {".tsx"},
    "yaml": {".yaml", ".yml"},
    "xml": {".xml", ".xsd", ".xsl", ".xslt"},
    "zig": {".zig"},
}

SPECIAL_FILE_LANGUAGE_MAP: Dict[str, str] = {
    "CMakeLists.txt": "cmake",
}

GLOBAL_SCOPE_NAME = "<global>"


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime options resolved from CLI arguments."""

    repo_path: Path | None
    repo_url: str | None
    local_repo_path: Path | None
    output_csv: Path
    max_commits: int | None
    languages: List[str]
    snapshot_mode: str = "current"
    enable_rca_metrics: bool = False
    rca_command: str = "rust-code-analysis-cli"
    rca_debug_dump_dir: Path | None = None
    skip_commits: int = 0
    log_level: str = "INFO"
