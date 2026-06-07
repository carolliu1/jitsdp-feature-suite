"""JavaScript method and call extraction using tree-sitter."""

from __future__ import annotations

from typing import List, Tuple

from tree_sitter import Node

from commit_feature_suite.models import CallSite, MethodInfo
from commit_feature_suite.parsers.parser_common import BaseLanguageParser


class JavaScriptParser(BaseLanguageParser):
    """Extract JavaScript functions, class methods, and call expressions."""

    language_name = "javascript"

    def _extract(self, root: Node, source_code: str, file_path: str) -> tuple[List[MethodInfo], List[CallSite]]:
        methods: List[MethodInfo] = []
        method_nodes: List[Tuple[MethodInfo, Node]] = []
        self._walk(root, source_code, file_path, "", methods, method_nodes)

        callsites: List[CallSite] = []
        for method, node in method_nodes:
            for call_node in self.nodes_by_types(node, {"call_expression"}):
                callee_name, receiver_name = self._extract_call_name(call_node, source_code)
                if callee_name:
                    callsites.append(
                        CallSite(
                            caller_method_id=method.method_id,
                            callee_name=callee_name,
                            receiver_name=receiver_name,
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
                if child.type == "class_declaration":
                    name_node = child.child_by_field_name("name")
                    class_name = self.node_text(name_node, source_code) if name_node else current_scope_class
                    stack.append((child, class_name))
                    continue

                if child.type == "function_declaration":
                    name_node = child.child_by_field_name("name")
                    if name_node is not None:
                        method = self.make_method(
                            file_path=file_path,
                            class_name=current_scope_class,
                            method_name=self.node_text(name_node, source_code),
                            node=child,
                        )
                        methods.append(method)
                        method_nodes.append((method, child))

                if child.type == "method_definition":
                    name_node = child.child_by_field_name("name")
                    if name_node is not None:
                        method = self.make_method(
                            file_path=file_path,
                            class_name=current_scope_class,
                            method_name=self.node_text(name_node, source_code),
                            node=child,
                        )
                        methods.append(method)
                        method_nodes.append((method, child))

                if child.type in {"lexical_declaration", "variable_declaration"}:
                    for declarator in self.nodes_by_types(child, {"variable_declarator"}):
                        name_node = declarator.child_by_field_name("name")
                        value_node = declarator.child_by_field_name("value")
                        if name_node is None or value_node is None:
                            continue
                        if value_node.type in {"function", "arrow_function"}:
                            method = self.make_method(
                                file_path=file_path,
                                class_name=current_scope_class,
                                method_name=self.node_text(name_node, source_code),
                                node=declarator,
                            )
                            methods.append(method)
                            method_nodes.append((method, value_node))

                stack.append((child, current_scope_class))

    def _extract_call_name(self, call_node: Node, source_code: str) -> tuple[str | None, str | None]:
        function_node = call_node.child_by_field_name("function")
        if function_node is None:
            return None, None
        if function_node.type == "identifier":
            return self.node_text(function_node, source_code), None
        if function_node.type == "member_expression":
            property_node = function_node.child_by_field_name("property")
            object_node = function_node.child_by_field_name("object")
            callee_name = self.node_text(property_node, source_code) if property_node else None
            receiver_name = self.node_text(object_node, source_code) if object_node else None
            return callee_name, receiver_name
        return None, None

