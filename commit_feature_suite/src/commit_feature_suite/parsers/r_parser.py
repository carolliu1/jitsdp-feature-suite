"""R function and call extraction using tree-sitter."""

from __future__ import annotations

from typing import List, Tuple

from tree_sitter import Node

from commit_feature_suite.models import CallSite, MethodInfo
from commit_feature_suite.parsers.parser_common import BaseLanguageParser


class RParser(BaseLanguageParser):
    """Extract R function assignments and call expressions."""

    language_name = "r"

    def _extract(self, root: Node, source_code: str, file_path: str) -> tuple[List[MethodInfo], List[CallSite]]:
        methods: List[MethodInfo] = []
        method_nodes: List[Tuple[MethodInfo, Node]] = []

        # R function declarations are typically assignments: name <- function(...) { ... }
        for assign in self.nodes_by_types(root, {"left_assignment", "equals_assignment", "right_assignment"}):
            left_node = assign.child_by_field_name("left")
            right_node = assign.child_by_field_name("right")
            if left_node is None or right_node is None:
                # Fallback for grammars not exposing field names.
                if len(assign.children) >= 3:
                    left_node = assign.children[0]
                    right_node = assign.children[-1]
            if left_node is None or right_node is None or right_node.type != "function_definition":
                continue

            method_name = self.last_identifier_text(left_node, source_code)
            if not method_name:
                continue
            method = self.make_method(
                file_path=file_path,
                class_name="",
                method_name=self.normalize_callable_name(method_name),
                node=right_node,
            )
            methods.append(method)
            method_nodes.append((method, right_node))

        callsites: List[CallSite] = []
        for method, node in method_nodes:
            for call_node in self.nodes_by_types(node, {"call"}):
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
                        caller_class_name="",
                        language=self.language_name,
                        dotted_callee=dotted_callee,
                    )
                )

        return methods, callsites

    def _extract_call_name(self, call_node: Node, source_code: str) -> tuple[str, str | None, str | None] | None:
        if not call_node.children:
            return None
        callee_node = call_node.children[0]
        dotted = self.node_text(callee_node, source_code)
        callee_name = self.last_identifier_text(callee_node, source_code)
        if not callee_name:
            return None
        name = self.normalize_callable_name(callee_name)
        receiver_name = None
        if "::" in dotted:
            receiver_name = dotted.rsplit("::", 1)[0].strip()
        elif "$" in dotted:
            receiver_name = dotted.rsplit("$", 1)[0].strip()
        return name, receiver_name, dotted

