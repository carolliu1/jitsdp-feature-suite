"""Python method and call extraction using tree-sitter."""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from tree_sitter import Node

from commit_feature_suite.config import GLOBAL_SCOPE_NAME
from commit_feature_suite.models import CallSite, MethodInfo, ParsedFile
from commit_feature_suite.parsers.parser_common import BaseLanguageParser


class PythonParser(BaseLanguageParser):
    """Extract Python functions, methods, imports, and call expressions."""

    language_name = "python"

    _IMPORT_RE = re.compile(r"^\s*import\s+([a-zA-Z0-9_.,\s]+)\s*$")
    _FROM_IMPORT_RE = re.compile(
        r"^\s*from\s+([a-zA-Z0-9_\.]+)\s+import\s+([a-zA-Z0-9_.,\s*]+)\s*$"
    )

    def parse_file(self, source_code: str, file_path: str) -> ParsedFile:
        """Parse Python file and provide richer metadata for cross-file resolution."""
        tree = self._parser.parse(source_code.encode("utf-8"))
        methods, callsites, metadata = self._extract_extended(tree.root_node, source_code, file_path)
        metadata = dict(metadata)
        metadata["_tree_root"] = tree.root_node
        return ParsedFile(methods=methods, callsites=callsites, metadata=metadata)

    def _extract(self, root: Node, source_code: str, file_path: str) -> tuple[List[MethodInfo], List[CallSite]]:
        """Compatibility implementation for BaseLanguageParser."""
        methods, callsites, _ = self._extract_extended(root, source_code, file_path)
        return methods, callsites

    def _extract_extended(
        self,
        root: Node,
        source_code: str,
        file_path: str,
    ) -> tuple[List[MethodInfo], List[CallSite], Dict[str, object]]:
        methods: List[MethodInfo] = []
        method_nodes: List[Tuple[MethodInfo, Node]] = []
        self._walk(root, source_code, file_path, GLOBAL_SCOPE_NAME, methods, method_nodes)

        imports = self._parse_imports(source_code)
        callsites: List[CallSite] = []
        for method, node in method_nodes:
            body = self.child_by_type(node, "block")
            if body is None:
                continue
            receiver_types = self._infer_receiver_types(body, source_code)
            for call_node in self.nodes_by_types(body, {"call"}):
                call_info = self._extract_call_name(call_node, source_code, receiver_types, method.class_name)
                if call_info is None:
                    continue
                callee_name, receiver_name, dotted_callee, inferred_receiver_type = call_info
                callsites.append(
                    CallSite(
                        caller_method_id=method.method_id,
                        callee_name=callee_name,
                        receiver_name=receiver_name,
                        caller_file_path=file_path,
                        caller_class_name=method.class_name,
                        language=self.language_name,
                        dotted_callee=dotted_callee,
                        inferred_receiver_type=inferred_receiver_type,
                    )
                )

        metadata: Dict[str, object] = {"imports": imports}
        return methods, callsites, metadata

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
                if child.type == "class_definition":
                    name_node = self.child_by_type(child, "identifier")
                    class_name = self.node_text(name_node, source_code) if name_node else current_scope_class
                    stack.append((child, class_name))
                    continue

                if child.type == "decorated_definition":
                    function_node = self.child_by_type(child, "function_definition")
                    if function_node is None:
                        stack.append((child, current_scope_class))
                        continue
                    name_node = self.child_by_type(function_node, "identifier")
                    if name_node is not None:
                        method = self.make_method(
                            file_path=file_path,
                            class_name=current_scope_class if current_scope_class != GLOBAL_SCOPE_NAME else "",
                            method_name=self.node_text(name_node, source_code),
                            node=function_node,
                        )
                        methods.append(method)
                        method_nodes.append((method, function_node))
                    stack.append((function_node, current_scope_class))
                    continue

                if child.type == "function_definition":
                    name_node = self.child_by_type(child, "identifier")
                    if name_node is not None:
                        method = self.make_method(
                            file_path=file_path,
                            class_name=current_scope_class if current_scope_class != GLOBAL_SCOPE_NAME else "",
                            method_name=self.node_text(name_node, source_code),
                            node=child,
                        )
                        methods.append(method)
                        method_nodes.append((method, child))

                stack.append((child, current_scope_class))

    def _extract_call_name(
        self,
        call_node: Node,
        source_code: str,
        receiver_types: Dict[str, str],
        current_class_name: str,
    ) -> tuple[str, str | None, str | None, str | None] | None:
        function_node = call_node.child_by_field_name("function")
        if function_node is None:
            return None

        if function_node.type == "identifier":
            name = self.node_text(function_node, source_code)
            return name, None, name, None

        if function_node.type == "attribute":
            attribute_node = function_node.child_by_field_name("attribute")
            object_node = function_node.child_by_field_name("object")
            if attribute_node is None:
                return None
            callee_name = self.node_text(attribute_node, source_code)
            receiver_name = self.node_text(object_node, source_code) if object_node else None
            inferred_type = None
            if receiver_name:
                if receiver_name == "self" and current_class_name:
                    inferred_type = current_class_name
                else:
                    inferred_type = receiver_types.get(receiver_name)
            dotted = self.node_text(function_node, source_code)
            return callee_name, receiver_name, dotted, inferred_type

        return None

    def _infer_receiver_types(self, body_node: Node, source_code: str) -> Dict[str, str]:
        """Infer simple local variable receiver types from constructor assignments."""
        receiver_types: Dict[str, str] = {}
        for assignment in self.nodes_by_types(body_node, {"assignment"}):
            left = assignment.child_by_field_name("left")
            right = assignment.child_by_field_name("right")
            if left is None or right is None or right.type != "call":
                continue
            constructor_node = right.child_by_field_name("function")
            if constructor_node is None:
                continue

            constructor_name = None
            if constructor_node.type == "identifier":
                constructor_name = self.node_text(constructor_node, source_code)
            elif constructor_node.type == "attribute":
                constructor_name = self.node_text(constructor_node, source_code)

            if constructor_name is None:
                continue

            if left.type == "identifier":
                receiver_types[self.node_text(left, source_code)] = constructor_name
        return receiver_types

    def _parse_imports(self, source_code: str) -> Dict[str, str]:
        """Parse import aliases for later cross-file call resolution."""
        imports: Dict[str, str] = {}
        for raw_line in source_code.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            import_match = self._IMPORT_RE.match(line)
            if import_match:
                modules = [item.strip() for item in import_match.group(1).split(",")]
                for module_part in modules:
                    if " as " in module_part:
                        module_name, alias = [item.strip() for item in module_part.split(" as ", 1)]
                        imports[alias] = module_name
                    else:
                        alias = module_part.split(".")[0]
                        imports[alias] = module_part
                continue

            from_match = self._FROM_IMPORT_RE.match(line)
            if from_match:
                module_name = from_match.group(1).strip()
                symbols = [item.strip() for item in from_match.group(2).split(",")]
                for symbol_part in symbols:
                    if symbol_part == "*":
                        continue
                    if " as " in symbol_part:
                        symbol_name, alias = [item.strip() for item in symbol_part.split(" as ", 1)]
                        imports[alias] = f"{module_name}.{symbol_name}"
                    else:
                        imports[symbol_part] = f"{module_name}.{symbol_part}"

        return imports

