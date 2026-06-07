"""Data models used across the project."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from commit_feature_suite.config import GLOBAL_SCOPE_NAME


@dataclass(frozen=True)
class MethodInfo:
    """Method-level node information stored in the call graph."""

    file_path: str
    class_name: str
    method_name: str
    start_line: int
    end_line: int
    language: str
    method_id: str = field(init=False)

    def __post_init__(self) -> None:
        class_name = self.class_name if self.class_name else GLOBAL_SCOPE_NAME
        method_id = f"{self.file_path}::{class_name}::{self.method_name}::{self.start_line}"
        object.__setattr__(self, "method_id", method_id)

    def to_node_attributes(self) -> Dict[str, object]:
        """Convert to node attributes for networkx."""
        return {
            "method_id": self.method_id,
            "file_path": self.file_path,
            "class_name": self.class_name,
            "method_name": self.method_name,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "language": self.language,
        }


@dataclass(frozen=True)
class CallSite:
    """A resolved-or-partially-resolved call extracted from a method body."""

    caller_method_id: str
    callee_name: str
    receiver_name: str | None = None
    caller_file_path: str | None = None
    caller_class_name: str | None = None
    language: str | None = None
    dotted_callee: str | None = None
    inferred_receiver_type: str | None = None


@dataclass
class ParsedFile:
    """Output from a language parser for one source file."""

    methods: List[MethodInfo]
    callsites: List[CallSite]
    metadata: Dict[str, object] = field(default_factory=dict)


