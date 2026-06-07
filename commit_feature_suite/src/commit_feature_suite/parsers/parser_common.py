"""Common tree-sitter parser utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Iterator, List

from tree_sitter import Node
from tree_sitter_language_pack import get_parser

from commit_feature_suite.models import CallSite, MethodInfo, ParsedFile


class BaseLanguageParser(ABC):
    """Base tree-sitter parser wrapper with shared traversal helpers."""

    language_name: str

    def __init__(self) -> None:
        self._parser = get_parser(self.language_name)

    def parse_file(self, source_code: str, file_path: str) -> ParsedFile:
        """Parse source code and return extracted methods and callsites."""
        tree = self._parser.parse(source_code.encode("utf-8"))
        methods, callsites = self._extract(tree.root_node, source_code, file_path)
        return ParsedFile(methods=methods, callsites=callsites, metadata={"_tree_root": tree.root_node})

    @abstractmethod
    def _extract(self, root: Node, source_code: str, file_path: str) -> tuple[List[MethodInfo], List[CallSite]]:
        """Extract method definitions and callsites from the AST."""

    @staticmethod
    def node_text(node: Node, source_code: str) -> str:
        """Return source text represented by a node."""
        return source_code[node.start_byte : node.end_byte]

    @staticmethod
    def child_by_type(node: Node, node_type: str) -> Node | None:
        """Return the first direct child with the given type."""
        for child in node.children:
            if child.type == node_type:
                return child
        return None

    @staticmethod
    def descendants(node: Node) -> Iterator[Node]:
        """Yield a node and all descendants without Python recursion."""
        stack = [node]
        while stack:
            current = stack.pop()
            yield current
            children = current.children
            if children:
                stack.extend(reversed(children))

    @staticmethod
    def nodes_by_types(node: Node, types: Iterable[str]) -> Iterator[Node]:
        """Yield descendant nodes whose type is in the given set."""
        type_set = set(types)
        for candidate in BaseLanguageParser.descendants(node):
            if candidate.type in type_set:
                yield candidate

    @staticmethod
    def line_number(node: Node) -> int:
        """Convert tree-sitter 0-based row to 1-based line number."""
        return node.start_point[0] + 1

    @staticmethod
    def end_line_number(node: Node) -> int:
        """Convert tree-sitter 0-based end row to 1-based line number."""
        return node.end_point[0] + 1

    def make_method(
        self,
        *,
        file_path: str,
        class_name: str,
        method_name: str,
        node: Node,
    ) -> MethodInfo:
        """Build a MethodInfo from an AST node."""
        return MethodInfo(
            file_path=file_path,
            class_name=class_name,
            method_name=method_name,
            start_line=self.line_number(node),
            end_line=self.end_line_number(node),
            language=self.language_name,
        )

    def last_identifier_text(self, node: Node, source_code: str) -> str | None:
        """Return the last identifier-like descendant text from node."""
        last_text: str | None = None
        for candidate in self.descendants(node):
            if candidate.type in {"identifier", "field_identifier", "property_identifier"}:
                last_text = self.node_text(candidate, source_code)
        return last_text

    @staticmethod
    def normalize_callable_name(text: str) -> str:
        """Normalize callable names by removing arguments/templates suffix when present."""
        name = text.strip()
        if "(" in name:
            name = name.split("(", 1)[0].strip()
        if "<" in name and ">" in name and "." not in name:
            name = name.split("<", 1)[0].strip()
        return name

