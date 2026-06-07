"""Rust method and call extraction using tree-sitter specialized rules."""

from __future__ import annotations

from typing import List, Tuple

from tree_sitter import Node

from commit_feature_suite.models import CallSite, MethodInfo
from commit_feature_suite.parsers.parser_common import BaseLanguageParser


class RustParser(BaseLanguageParser):
    """Extract Rust function definitions and callsites with receiver hints."""

    language_name = "rust"

    def _extract(self, root: Node, source_code: str, file_path: str) -> tuple[List[MethodInfo], List[CallSite]]:
        methods: List[MethodInfo] = []
        method_nodes: List[Tuple[MethodInfo, Node]] = []

        for node in self.nodes_by_types(root, {"function_item"}):
            method_name = self._extract_function_name(node, source_code)
            if not method_name:
                continue
            class_name = self._extract_owner_type(node, source_code)
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
            body = node.child_by_field_name("body") or self.child_by_type(node, "block")
            if body is None:
                continue

            for call_node in self.nodes_by_types(body, {"call_expression", "method_call_expression"}):
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

    def _extract_function_name(self, function_node: Node, source_code: str) -> str | None:
        name_node = function_node.child_by_field_name("name")
        if name_node is None:
            return None
        name = self.node_text(name_node, source_code).strip()
        if not name:
            return None
        return self.normalize_callable_name(name)

    def _extract_owner_type(self, function_node: Node, source_code: str) -> str:
        parent = function_node.parent
        while parent is not None:
            if parent.type == "impl_item":
                impl_type = parent.child_by_field_name("type")
                if impl_type is not None:
                    return self._normalize_rust_type(self.node_text(impl_type, source_code))
                ident = self.last_identifier_text(parent, source_code)
                return self._normalize_rust_type(ident or "")
            if parent.type == "trait_item":
                name_node = parent.child_by_field_name("name")
                if name_node is not None:
                    return self._normalize_rust_type(self.node_text(name_node, source_code))
                ident = self.last_identifier_text(parent, source_code)
                return self._normalize_rust_type(ident or "")
            parent = parent.parent
        return ""

    def _extract_call_name(self, call_node: Node, source_code: str) -> tuple[str, str | None, str | None] | None:
        if call_node.type == "method_call_expression":
            name_node = call_node.child_by_field_name("name")
            receiver_node = call_node.child_by_field_name("receiver")
            if name_node is None:
                name_text = self.last_identifier_text(call_node, source_code)
            else:
                name_text = self.node_text(name_node, source_code)
            if not name_text:
                return None
            callee_name = self.normalize_callable_name(name_text)
            receiver_name = self.node_text(receiver_node, source_code).strip() if receiver_node is not None else None
            dotted = self.node_text(call_node, source_code)
            return callee_name, receiver_name, dotted

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
        if "::" in dotted:
            receiver_name = dotted.rsplit("::", 1)[0].strip()
        elif "." in dotted:
            receiver_name = dotted.rsplit(".", 1)[0].strip()

        return callee_name, receiver_name, dotted

    @staticmethod
    def _normalize_rust_type(type_text: str) -> str:
        text = (type_text or "").strip()
        if not text:
            return ""
        text = text.replace("&", "").replace("mut ", "").strip()
        text = text.split("<", 1)[0].strip()
        if "::" in text:
            text = text.split("::")[-1].strip()
        return text

