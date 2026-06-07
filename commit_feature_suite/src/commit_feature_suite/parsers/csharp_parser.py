"""C# method and call extraction using tree-sitter."""

from __future__ import annotations

from typing import List, Tuple

from tree_sitter import Node

from commit_feature_suite.models import CallSite, MethodInfo
from commit_feature_suite.parsers.parser_common import BaseLanguageParser


class CSharpParser(BaseLanguageParser):
    """Extract C# method definitions and invocation expressions."""

    # tree-sitter-language-pack uses c_sharp for C#
    language_name = "c_sharp"

    def _extract(self, root: Node, source_code: str, file_path: str) -> tuple[List[MethodInfo], List[CallSite]]:
        methods: List[MethodInfo] = []
        method_nodes: List[Tuple[MethodInfo, Node]] = []
        self._walk(root, source_code, file_path, "", methods, method_nodes)

        callsites: List[CallSite] = []
        for method, node in method_nodes:
            for call_node in self.nodes_by_types(node, {"invocation_expression"}):
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
                        language="csharp",
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
                if child.type in {"class_declaration", "struct_declaration", "record_declaration"}:
                    name_node = child.child_by_field_name("name")
                    class_name = self.node_text(name_node, source_code) if name_node else current_scope_class
                    stack.append((child, class_name))
                    continue

                if child.type in {"method_declaration", "constructor_declaration", "local_function_statement"}:
                    name_node = child.child_by_field_name("name")
                    if name_node is None:
                        name_node = self.child_by_type(child, "identifier")
                    if name_node is not None:
                        method = MethodInfo(
                            file_path=file_path,
                            class_name=current_scope_class,
                            method_name=self.normalize_callable_name(self.node_text(name_node, source_code)),
                            start_line=self.line_number(child),
                            end_line=self.end_line_number(child),
                            language="csharp",
                        )
                        methods.append(method)
                        method_nodes.append((method, child))

                stack.append((child, current_scope_class))

    def _extract_call_name(self, call_node: Node, source_code: str) -> tuple[str, str | None, str | None] | None:
        function_node = call_node.child_by_field_name("function")
        if function_node is None and call_node.children:
            function_node = call_node.children[0]
        if function_node is None:
            return None

        dotted = self.node_text(function_node, source_code)
        callee_name = self.last_identifier_text(function_node, source_code)
        if not callee_name:
            return None
        callee_name = self.normalize_callable_name(callee_name)

        receiver_name = None
        if "." in dotted:
            receiver_name = dotted.rsplit(".", 1)[0].strip()
        return callee_name, receiver_name, dotted

