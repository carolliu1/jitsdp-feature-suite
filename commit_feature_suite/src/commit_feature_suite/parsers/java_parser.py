"""Java method and call extraction using tree-sitter."""

from __future__ import annotations

from typing import List, Tuple

from tree_sitter import Node

from commit_feature_suite.models import CallSite, MethodInfo
from commit_feature_suite.parsers.parser_common import BaseLanguageParser


class JavaParser(BaseLanguageParser):
    """Extract Java methods, constructors, and invocation expressions."""

    language_name = "java"

    def _extract(self, root: Node, source_code: str, file_path: str) -> tuple[List[MethodInfo], List[CallSite]]:
        methods: List[MethodInfo] = []
        method_nodes: List[Tuple[MethodInfo, Node]] = []
        self._walk(root, source_code, file_path, "", methods, method_nodes)

        callsites: List[CallSite] = []
        for method, node in method_nodes:
            body = self.child_by_type(node, "block")
            if body is None:
                continue
            for call_node in self.nodes_by_types(body, {"method_invocation"}):
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
                if child.type in {"class_declaration", "interface_declaration", "enum_declaration"}:
                    name_node = child.child_by_field_name("name")
                    class_name = self.node_text(name_node, source_code) if name_node else current_scope_class
                    stack.append((child, class_name))
                    continue

                if child.type in {"method_declaration", "constructor_declaration"}:
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

                stack.append((child, current_scope_class))

    def _extract_call_name(self, call_node: Node, source_code: str) -> tuple[str | None, str | None]:
        name_node = call_node.child_by_field_name("name")
        object_node = call_node.child_by_field_name("object")
        callee_name = self.node_text(name_node, source_code) if name_node else None
        receiver_name = self.node_text(object_node, source_code) if object_node else None
        return callee_name, receiver_name

