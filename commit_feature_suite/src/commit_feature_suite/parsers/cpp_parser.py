"""C++ method and call extraction using tree-sitter."""

from __future__ import annotations

from typing import List, Tuple

from tree_sitter import Node

from commit_feature_suite.models import CallSite, MethodInfo
from commit_feature_suite.parsers.parser_common import BaseLanguageParser


class CppParser(BaseLanguageParser):
    """Extract C++ function/method definitions and call expressions."""

    language_name = "cpp"
    def _extract(self, root: Node, source_code: str, file_path: str) -> tuple[List[MethodInfo], List[CallSite]]:
        methods: List[MethodInfo] = []
        method_nodes: List[Tuple[MethodInfo, Node]] = []

        for node in self.nodes_by_types(root, {"function_definition"}):
            declarator = node.child_by_field_name("declarator") or self.child_by_type(node, "function_declarator")
            if declarator is None:
                continue
            class_name, method_name = self._extract_scoped_function_name(declarator, source_code)
            if not method_name:
                continue
            method = self.make_method(
                file_path=file_path,
                class_name=class_name,
                method_name=method_name,
                node=node,
            )
            methods.append(method)
            method_nodes.append((method, node))

        callsites: List[CallSite] = []
        for method, node in method_nodes:
            body = node.child_by_field_name("body") or self.child_by_type(node, "compound_statement")
            if body is None:
                continue
            for call_node in self.nodes_by_types(body, {"call_expression"}):
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
                        language=self.language_name,
                        dotted_callee=dotted_callee,
                    )
                )

        return methods, callsites

    def _extract_scoped_function_name(self, declarator: Node, source_code: str) -> tuple[str, str]:
        raw = self.node_text(declarator, source_code)
        raw = self.normalize_callable_name(raw)
        if "::" in raw:
            parts = [part.strip() for part in raw.split("::") if part.strip()]
            if len(parts) >= 2:
                return "::".join(parts[:-1]), parts[-1]

        identifier = self.last_identifier_text(declarator, source_code)
        if not identifier:
            return "", ""
        return "", self.normalize_callable_name(identifier)

    def _extract_call_name(self, call_node: Node, source_code: str) -> tuple[str, str | None, str | None] | None:
        function_node = call_node.child_by_field_name("function")
        if function_node is None and call_node.children:
            function_node = call_node.children[0]
        if function_node is None:
            return None

        dotted = self.node_text(function_node, source_code)
        name = self.last_identifier_text(function_node, source_code)
        if not name:
            return None
        callee_name = self.normalize_callable_name(name)

        receiver_name = None
        if "->" in dotted:
            receiver_name = dotted.split("->", 1)[0].strip()
        elif "." in dotted:
            receiver_name = dotted.rsplit(".", 1)[0].strip()
        elif "::" in dotted:
            receiver_name = dotted.rsplit("::", 1)[0].strip()

        return callee_name, receiver_name, dotted

