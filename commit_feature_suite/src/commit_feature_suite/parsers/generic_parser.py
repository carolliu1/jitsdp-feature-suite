"""Generic tree-sitter parser for additional mainstream languages."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Set, Tuple

from tree_sitter import Node

from commit_feature_suite.config import GLOBAL_SCOPE_NAME
from commit_feature_suite.models import CallSite, MethodInfo, ParsedFile
from commit_feature_suite.parsers.language_profiles import LanguageProfile
from commit_feature_suite.parsers.parser_common import BaseLanguageParser


class GenericTreeSitterParser(BaseLanguageParser):
    """Heuristic method/call extractor for languages without dedicated parsers."""

    _CLASS_NODE_TYPES: Set[str] = {
        "class_declaration",
        "class_definition",
        "struct_declaration",
        "interface_declaration",
        "record_declaration",
        "enum_declaration",
        "object_declaration",
    }
    _METHOD_NODE_TYPES: Set[str] = {
        "function_definition",
        "function_declaration",
        "method_definition",
        "method_declaration",
        "constructor_declaration",
        "local_function_statement",
        "function_item",
        "function",
    }
    _CALL_NODE_TYPES: Set[str] = {
        "call",
        "call_expression",
        "invocation_expression",
        "method_invocation",
        "function_call_expression",
    }
    _DEFAULT_IMPORT_PATTERNS: List[re.Pattern[str]] = []
    _DEFAULT_RECEIVER_TYPE_PATTERNS: List[re.Pattern[str]] = [
        re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*new\s+([A-Z][A-Za-z0-9_]*)"),
        re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*([A-Z][A-Za-z0-9_<>]*)\s*="),
        re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*:=\s*([A-Z][A-Za-z0-9_]*)\s*\{"),
        re.compile(r"\blet\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([A-Z][A-Za-z0-9_:]*)::"),
    ]

    def __init__(
        self,
        language_name: str | None = None,
        *,
        profile: LanguageProfile | None = None,
        class_node_types: Iterable[str] | None = None,
        method_node_types: Iterable[str] | None = None,
        call_node_types: Iterable[str] | None = None,
        import_patterns: Iterable[str] | None = None,
        receiver_type_patterns: Iterable[str] | None = None,
    ) -> None:
        if profile is not None:
            self.language_name = profile.tree_sitter_language
            self._profile_language = profile.language
            if profile.class_node_types:
                self._CLASS_NODE_TYPES = set(profile.class_node_types)
            if profile.method_node_types:
                self._METHOD_NODE_TYPES = set(profile.method_node_types)
            if profile.call_node_types:
                self._CALL_NODE_TYPES = set(profile.call_node_types)
        else:
            if language_name is None:
                raise ValueError("language_name is required when profile is not provided")
            self.language_name = language_name
            self._profile_language = language_name
        if class_node_types is not None:
            self._CLASS_NODE_TYPES = set(class_node_types)
        if method_node_types is not None:
            self._METHOD_NODE_TYPES = set(method_node_types)
        if call_node_types is not None:
            self._CALL_NODE_TYPES = set(call_node_types)
        if import_patterns is not None:
            self._import_patterns = [re.compile(pattern) for pattern in import_patterns]
        else:
            self._import_patterns = list(self._DEFAULT_IMPORT_PATTERNS)
        if receiver_type_patterns is not None:
            self._receiver_type_patterns = [re.compile(pattern) for pattern in receiver_type_patterns]
        else:
            self._receiver_type_patterns = list(self._DEFAULT_RECEIVER_TYPE_PATTERNS)
        super().__init__()

    def _extract(self, root: Node, source_code: str, file_path: str) -> tuple[List[MethodInfo], List[CallSite]]:
        parsed = self.parse_file(source_code, file_path)
        return parsed.methods, parsed.callsites

    def parse_file(self, source_code: str, file_path: str) -> ParsedFile:
        """Parse source and include import/type metadata for semantic resolution."""
        tree = self._parser.parse(source_code.encode("utf-8"))
        methods: List[MethodInfo] = []
        method_nodes: List[Tuple[MethodInfo, Node]] = []
        self._walk(tree.root_node, source_code, file_path, GLOBAL_SCOPE_NAME, methods, method_nodes)

        imports = self._parse_imports(source_code)
        callsites: List[CallSite] = []
        for method, node in method_nodes:
            receiver_types = self._infer_receiver_types(node, source_code)
            for call_node in self.nodes_by_types(node, self._CALL_NODE_TYPES):
                call_info = self._extract_call_name(call_node, source_code)
                if call_info is None:
                    continue
                callee_name, receiver_name, dotted_callee = call_info
                inferred_receiver_type = receiver_types.get(receiver_name) if receiver_name else None
                callsites.append(
                    CallSite(
                        caller_method_id=method.method_id,
                        callee_name=callee_name,
                        receiver_name=receiver_name,
                        caller_file_path=file_path,
                        caller_class_name=method.class_name,
                        language=self._profile_language,
                        dotted_callee=dotted_callee,
                        inferred_receiver_type=inferred_receiver_type,
                    )
                )
        return ParsedFile(methods=methods, callsites=callsites, metadata={"imports": imports, "_tree_root": tree.root_node})

    def _extract_legacy(self, root: Node, source_code: str, file_path: str) -> tuple[List[MethodInfo], List[CallSite]]:
        methods: List[MethodInfo] = []
        method_nodes: List[Tuple[MethodInfo, Node]] = []
        self._walk(root, source_code, file_path, GLOBAL_SCOPE_NAME, methods, method_nodes)

        callsites: List[CallSite] = []
        for method, node in method_nodes:
            for call_node in self.nodes_by_types(node, self._CALL_NODE_TYPES):
                call_info = self._extract_call_name(call_node, source_code)
                if call_info is None:
                    continue
                callee_name, receiver_name, dotted_callee = call_info
                callsites.append(
                    CallSite(
                        caller_method_id=method.method_id,
                        callee_name=callee_name,
                        receiver_name=receiver_name,
                        caller_file_path=file_path,
                        caller_class_name=method.class_name,
                        language=self._profile_language,
                        dotted_callee=dotted_callee,
                    )
                )
        return methods, callsites

    def _walk(
        self,
        node: Node,
        source_code: str,
        file_path: str,
        current_class: str,
        methods: List[MethodInfo],
        method_nodes: List[Tuple[MethodInfo, Node]],
    ) -> None:
        stack: List[Tuple[Node, str]] = [(node, current_class)]
        while stack:
            current_node, current_scope_class = stack.pop()
            for child in reversed(current_node.children):
                if child.type in self._CLASS_NODE_TYPES:
                    class_name = self._extract_name(child, source_code) or current_scope_class
                    stack.append((child, class_name))
                    continue

                if child.type in self._METHOD_NODE_TYPES:
                    method_name = self._extract_name(child, source_code)
                    if method_name:
                        method = self.make_method(
                            file_path=file_path,
                            class_name=current_scope_class if current_scope_class != GLOBAL_SCOPE_NAME else "",
                            method_name=self.normalize_callable_name(method_name),
                            node=child,
                        )
                        methods.append(method)
                        method_nodes.append((method, child))

                stack.append((child, current_scope_class))

    def _extract_name(self, node: Node, source_code: str) -> str | None:
        name_node = node.child_by_field_name("name")
        if name_node is not None:
            return self.node_text(name_node, source_code)

        identifier = self.last_identifier_text(node, source_code)
        if identifier:
            return identifier
        return None

    def _extract_call_name(self, call_node: Node, source_code: str) -> tuple[str, str | None, str | None] | None:
        function_node = (
            call_node.child_by_field_name("function")
            or call_node.child_by_field_name("name")
            or call_node.child_by_field_name("callee")
        )
        if function_node is None and call_node.children:
            function_node = call_node.children[0]
        if function_node is None:
            return None

        dotted = self.node_text(function_node, source_code)
        callee_name = self.last_identifier_text(function_node, source_code)
        if not callee_name:
            return None

        name = self.normalize_callable_name(callee_name)
        receiver_name = None
        for token in ("->", "::", ".", "$"):
            if token in dotted:
                receiver_name = dotted.rsplit(token, 1)[0].strip()
                break
        return name, receiver_name, dotted

    def _parse_imports(self, source_code: str) -> Dict[str, str]:
        imports: Dict[str, str] = {}
        patterns = self._import_patterns
        if not patterns:
            return imports

        for raw_line in source_code.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("//") or line.startswith("#"):
                continue
            for pattern in patterns:
                match = pattern.match(line)
                if not match:
                    continue
                groups = match.groups()
                if len(groups) == 1:
                    symbol = groups[0].strip()
                    imports[symbol] = symbol
                elif len(groups) == 2 and "{" in line:
                    for part in groups[1].split(","):
                        item = part.strip()
                        if not item:
                            continue
                        alias = item.split(" as ")[-1].strip()
                        original = item.split(" as ")[0].strip()
                        imports[alias] = original
                elif len(groups) == 1 and "\"" in line:
                    module = groups[0]
                    alias = module.split("/")[-1]
                    imports[alias] = module
                elif len(groups) == 2 and "\"" in line:
                    alias, module = groups
                    imports[alias] = module
                elif len(groups) == 2 and "require" in line:
                    imports[groups[1].split("/")[-1]] = groups[1]
                else:
                    symbol = groups[0].split(".")[-1].split("::")[-1].split("\\")[-1]
                    imports[symbol] = groups[0]
        return imports

    def _infer_receiver_types(self, method_node: Node, source_code: str) -> Dict[str, str]:
        receiver_types: Dict[str, str] = {}
        method_text = self.node_text(method_node, source_code)
        for pattern in self._receiver_type_patterns:
            for match in pattern.finditer(method_text):
                receiver, inferred_type = match.groups()
                cleaned_type = inferred_type.split("<")[0].split("::")[-1].strip()
                if cleaned_type:
                    receiver_types[receiver] = cleaned_type
        return receiver_types

