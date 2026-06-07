"""Global variable reference counter for affected methods."""

from __future__ import annotations

from typing import Dict, Iterable, List, Set

from tree_sitter import Node
from tree_sitter_language_pack import get_parser

from commit_feature_suite.models import MethodInfo


FUNCTION_LIKE_NODE_TYPES: Set[str] = {
    "function_definition",
    "function_declaration",
    "function_item",
    "method_definition",
    "method_declaration",
    "method",
    "constructor_declaration",
    "lambda_expression",
    "arrow_function",
}

IDENTIFIER_NODE_TYPES: Set[str] = {"identifier", "field_identifier", "property_identifier"}

PARAMETER_NODE_TYPES: Set[str] = {
    "parameter",
    "formal_parameter",
    "parameter_declaration",
    "required_parameter",
    "optional_parameter",
    "default_parameter",
    "typed_parameter",
}

LOCAL_DECL_NODE_TYPES: Set[str] = {
    "variable_declarator",
    "init_declarator",
    "local_variable_declaration",
    "lexical_declaration",
    "variable_declaration",
    "assignment",
    "for_statement",
    "for_in_clause",
    "catch_clause",
    "with_item",
}

SUPPORTED_SPECIALIZED_LANGUAGES: Set[str] = {"c", "cpp", "csharp", "python", "java", "javascript", "typescript", "tsx"}

TREE_SITTER_LANGUAGE_ALIASES: Dict[str, str] = {
    "csharp": "c_sharp",
    "objectivec": "objc",
    "lisp": "commonlisp",
    "shell": "bash",
    "tsx": "typescript",
}


class GlobalVariableCounter:
    """Count method-level global variable references with language-tuned heuristics."""

    def __init__(self, logger) -> None:
        self.logger = logger
        self._parsers: Dict[str, object] = {}

    def count_for_methods(self, *, language: str, source_code: str, methods: Iterable[MethodInfo]) -> Dict[str, int]:
        parser = self._get_parser(language)
        method_list = list(methods)
        if parser is None:
            return {method.method_id: 0 for method in method_list}

        root = parser.parse(source_code.encode("utf-8")).root_node
        return self.count_for_methods_with_root(
            language=language,
            source_code=source_code,
            methods=method_list,
            root=root,
        )

    def count_for_methods_with_root(self, *, language: str, source_code: str, methods: Iterable[MethodInfo], root: Node) -> Dict[str, int]:
        method_list = list(methods)
        file_scope_symbols = self._collect_file_scope_symbols(root, source_code, language)
        class_scope_symbols = self._collect_class_scope_symbols(root, source_code, language)
        result: Dict[str, int] = {}
        for method in method_list:
            method_node = self._best_method_node(root, method)
            if method_node is None:
                result[method.method_id] = 0
                continue

            referenced = self._collect_variable_like_references(method_node, source_code)
            local_symbols = self._collect_local_symbols(method_node, source_code)
            python_global_symbols = self._collect_python_global_symbols(method_node, source_code) if language == "python" else set()

            if language in SUPPORTED_SPECIALIZED_LANGUAGES:
                candidate_globals = (file_scope_symbols | class_scope_symbols | python_global_symbols) - local_symbols
                count = len(referenced & candidate_globals)
            else:
                count = len(referenced - local_symbols)
            result[method.method_id] = max(0, count)
        return result

    def _get_parser(self, language: str):
        parser_language = TREE_SITTER_LANGUAGE_ALIASES.get(language, language)
        if parser_language in self._parsers:
            return self._parsers[parser_language]
        try:
            parser = get_parser(parser_language)
            self._parsers[parser_language] = parser
            return parser
        except Exception:
            self._parsers[parser_language] = None
            self.logger.debug(
                "No tree-sitter parser for global-var language: project_language=%s parser_language=%s",
                language,
                parser_language,
            )
            return None

    def _best_method_node(self, root: Node, method: MethodInfo) -> Node | None:
        start_row = method.start_line - 1
        end_row = method.end_line - 1
        candidates: List[Node] = []
        for node in self._descendants(root):
            if node.type not in FUNCTION_LIKE_NODE_TYPES:
                continue
            if node.start_point[0] <= start_row and node.end_point[0] >= end_row:
                candidates.append(node)
        if not candidates:
            return None
        candidates.sort(key=lambda node: node.end_byte - node.start_byte)
        return candidates[0]

    def _collect_variable_like_references(self, node: Node, source_code: str) -> Set[str]:
        refs: Set[str] = set()
        for ident in self._descendants(node):
            if ident.type not in IDENTIFIER_NODE_TYPES:
                continue
            if not self._is_reference_like(ident):
                continue
            text = source_code[ident.start_byte : ident.end_byte].strip()
            if text:
                refs.add(text)
        return refs

    def _is_reference_like(self, ident: Node) -> bool:
        parent = ident.parent
        if parent is None:
            return False
        parent_type = parent.type
        # Skip declaration/name positions that are not variable reads/writes.
        if parent_type in {
            "function_definition",
            "function_declaration",
            "method_definition",
            "method_declaration",
            "class_definition",
            "class_declaration",
            "type_identifier",
            "import_statement",
            "package_declaration",
            "namespace_declaration",
        }:
            return False
        return True

    def _collect_local_symbols(self, method_node: Node, source_code: str) -> Set[str]:
        local_symbols: Set[str] = set()
        for node in self._descendants(method_node):
            if node.type in PARAMETER_NODE_TYPES or node.type in LOCAL_DECL_NODE_TYPES:
                for ident in self._descendants(node):
                    if ident.type in IDENTIFIER_NODE_TYPES:
                        text = source_code[ident.start_byte : ident.end_byte].strip()
                        if text:
                            local_symbols.add(text)
        return local_symbols

    def _collect_python_global_symbols(self, method_node: Node, source_code: str) -> Set[str]:
        symbols: Set[str] = set()
        for node in self._descendants(method_node):
            if node.type != "global_statement":
                continue
            for ident in self._descendants(node):
                if ident.type == "identifier":
                    text = source_code[ident.start_byte : ident.end_byte].strip()
                    if text:
                        symbols.add(text)
        return symbols

    def _collect_file_scope_symbols(self, root: Node, source_code: str, language: str) -> Set[str]:
        symbols: Set[str] = set()
        for node in root.children:
            if node.type in FUNCTION_LIKE_NODE_TYPES:
                continue
            if node.type in {"class_definition", "class_declaration"}:
                continue
            if node.type in {
                "declaration",
                "declaration_list",
                "field_declaration",
                "variable_declaration",
                "lexical_declaration",
                "assignment",
                "expression_statement",
            }:
                symbols.update(self._identifiers_in_declaration_like(node, source_code))
            if language == "python" and node.type in {"assignment", "annotated_assignment"}:
                symbols.update(self._identifiers_in_declaration_like(node, source_code))
        return symbols

    def _collect_class_scope_symbols(self, root: Node, source_code: str, language: str) -> Set[str]:
        if language not in {"java", "csharp", "javascript", "typescript", "tsx", "cpp"}:
            return set()
        symbols: Set[str] = set()
        for node in self._descendants(root):
            if node.type not in {"class_definition", "class_declaration"}:
                continue
            for child in node.children:
                if child.type in {"field_declaration", "property_signature", "public_field_definition", "variable_declaration"}:
                    symbols.update(self._identifiers_in_declaration_like(child, source_code))
        return symbols

    def _identifiers_in_declaration_like(self, node: Node, source_code: str) -> Set[str]:
        out: Set[str] = set()
        for child in self._descendants(node):
            if child.type in IDENTIFIER_NODE_TYPES:
                text = source_code[child.start_byte : child.end_byte].strip()
                if text:
                    out.add(text)
        return out

    def _descendants(self, node: Node):
        stack = [node]
        while stack:
            current = stack.pop()
            yield current
            children = current.children
            if children:
                stack.extend(reversed(children))

