"""Call edge resolution logic."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx

from commit_feature_suite.graph.indexes import GraphIndexes, python_module_name
from commit_feature_suite.models import CallSite, MethodInfo


class CallGraphResolver:
    """Resolve callsites to graph edges using local + semantic + fallback matching."""

    def __init__(self, logger) -> None:
        self.logger = logger

    def add_edges(
        self,
        *,
        graph: nx.DiGraph,
        methods_by_file: Dict[str, List[MethodInfo]],
        methods_by_id: Dict[str, MethodInfo],
        callsites_by_file: Dict[str, List[CallSite]],
        file_metadata: Dict[str, Dict[str, object]],
        indexes: GraphIndexes,
    ) -> None:
        """Resolve all callsites and add caller->callee edges."""
        for file_path, callsites in callsites_by_file.items():
            imports = self.extract_imports_for_file(file_metadata.get(file_path, {}))
            for callsite in callsites:
                caller = self.resolve_caller(callsite, methods_by_id, methods_by_file)
                if caller is None:
                    continue
                callee = self.resolve_callee(
                    caller=caller,
                    callsite=callsite,
                    imports=imports,
                    indexes=indexes,
                )
                if callee is None:
                    continue
                graph.add_edge(caller.method_id, callee.method_id)

    def resolve_callee(
        self,
        *,
        caller: MethodInfo,
        callsite: CallSite,
        imports: Dict[str, str],
        indexes: GraphIndexes,
    ) -> MethodInfo | None:
        """Resolve a call target with local-first and cross-file fallback."""
        if caller.class_name:
            same_class = indexes.methods_by_name_and_class_and_file.get(
                (caller.file_path, caller.class_name, callsite.callee_name),
                [],
            )
            if same_class:
                return self.choose_best_candidate(same_class, caller.start_line)

        same_file = indexes.methods_by_name_and_file.get((caller.file_path, callsite.callee_name), [])
        if same_file:
            return self.choose_best_candidate(same_file, caller.start_line)

        semantic_match = self.resolve_semantic_cross_file(
            caller=caller,
            callsite=callsite,
            imports=imports,
            indexes=indexes,
        )
        if semantic_match is not None:
            return semantic_match

        if caller.language == "python":
            py_match = self.resolve_python_cross_file(
                caller=caller,
                callsite=callsite,
                imports=imports,
                indexes=indexes,
            )
            if py_match is not None:
                return py_match

        global_name = indexes.methods_by_name_global.get(callsite.callee_name, [])
        if len(global_name) == 1:
            return global_name[0]
        return None

    def resolve_semantic_cross_file(
        self,
        *,
        caller: MethodInfo,
        callsite: CallSite,
        imports: Dict[str, str],
        indexes: GraphIndexes,
    ) -> MethodInfo | None:
        """Generic semantic resolution using inferred receiver types and imports."""
        if callsite.inferred_receiver_type:
            inferred_type = callsite.inferred_receiver_type.split(".")[-1].split("::")[-1].split("\\")[-1]
            class_hits = indexes.methods_by_class_and_name_global.get((inferred_type, callsite.callee_name), [])
            if len(class_hits) == 1:
                return class_hits[0]
            if class_hits:
                return self.choose_best_candidate(class_hits, caller.start_line)

        if callsite.receiver_name and callsite.receiver_name in imports:
            imported_ref = imports[callsite.receiver_name]
            imported_symbol = imported_ref.split(".")[-1].split("::")[-1].split("\\")[-1].split("/")[-1]
            class_hits = indexes.methods_by_class_and_name_global.get((imported_symbol, callsite.callee_name), [])
            if class_hits:
                return self.choose_best_candidate(class_hits, caller.start_line)

        if callsite.callee_name in imports:
            imported_ref = imports[callsite.callee_name]
            imported_symbol = imported_ref.split(".")[-1].split("::")[-1].split("\\")[-1].split("/")[-1]
            direct_hits = indexes.methods_by_class_and_name_global.get((imported_symbol, callsite.callee_name), [])
            if direct_hits:
                return self.choose_best_candidate(direct_hits, caller.start_line)

        if caller.language == "python":
            module_name = python_module_name(caller.file_path)
            module_hits = indexes.methods_by_python_module_and_name.get((module_name, callsite.callee_name), [])
            if module_hits:
                return self.choose_best_candidate(module_hits, caller.start_line)
            if caller.class_name:
                class_hits = indexes.methods_by_python_module_class_and_name.get(
                    (module_name, caller.class_name, callsite.callee_name),
                    [],
                )
                if class_hits:
                    return self.choose_best_candidate(class_hits, caller.start_line)
        return None

    def resolve_caller(
        self,
        callsite: CallSite,
        methods_by_id: Dict[str, MethodInfo],
        methods_by_file: Dict[str, List[MethodInfo]],
    ) -> MethodInfo | None:
        """Resolve caller method from direct method id or fuzzy parse fallback."""
        caller = methods_by_id.get(callsite.caller_method_id)
        if caller is not None:
            return caller

        parsed = self.parse_method_id(callsite.caller_method_id)
        if parsed is None:
            return None
        caller_file, caller_class, caller_name, caller_start_line = parsed
        file_methods = methods_by_file.get(caller_file, [])
        candidates = [
            method
            for method in file_methods
            if method.method_name == caller_name and method.class_name == caller_class
        ]
        if not candidates:
            candidates = [method for method in file_methods if method.method_name == caller_name]
        if not candidates:
            return None
        return self.choose_best_candidate(candidates, caller_start_line)

    def resolve_python_cross_file(
        self,
        *,
        caller: MethodInfo,
        callsite: CallSite,
        imports: Dict[str, str],
        indexes: GraphIndexes,
    ) -> MethodInfo | None:
        """Resolve Python calls using inferred receiver types and imports."""
        if callsite.inferred_receiver_type:
            receiver_type = callsite.inferred_receiver_type.split(".")[-1]
            class_hits = indexes.methods_by_class_and_name_global.get((receiver_type, callsite.callee_name), [])
            if len(class_hits) == 1:
                return class_hits[0]
            if class_hits:
                caller_module = python_module_name(caller.file_path)
                same_module_hits = [
                    item for item in class_hits if python_module_name(item.file_path) == caller_module
                ]
                if same_module_hits:
                    return self.choose_best_candidate(same_module_hits, caller.start_line)

        if callsite.receiver_name and callsite.receiver_name in imports:
            imported_ref = imports[callsite.receiver_name]
            module_name, class_name = self.split_module_and_symbol(imported_ref)
            if class_name:
                matches = indexes.methods_by_python_module_class_and_name.get(
                    (module_name, class_name, callsite.callee_name),
                    [],
                )
                if matches:
                    return self.choose_best_candidate(matches, caller.start_line)

        if callsite.dotted_callee and "." in callsite.dotted_callee:
            receiver_alias = callsite.dotted_callee.split(".", 1)[0]
            imported_ref = imports.get(receiver_alias)
            if imported_ref:
                module_name, class_name = self.split_module_and_symbol(imported_ref)
                if class_name:
                    matches = indexes.methods_by_python_module_class_and_name.get(
                        (module_name, class_name, callsite.callee_name),
                        [],
                    )
                else:
                    matches = indexes.methods_by_python_module_and_name.get((module_name, callsite.callee_name), [])
                if matches:
                    return self.choose_best_candidate(matches, caller.start_line)

        direct_import = imports.get(callsite.callee_name)
        if direct_import:
            module_name, imported_symbol = self.split_module_and_symbol(direct_import)
            if imported_symbol:
                func_hits = indexes.methods_by_python_module_and_name.get((module_name, imported_symbol), [])
                if func_hits:
                    return self.choose_best_candidate(func_hits, caller.start_line)
                class_hits = indexes.methods_by_python_module_class_and_name.get(
                    (module_name, imported_symbol, callsite.callee_name),
                    [],
                )
                if class_hits:
                    return self.choose_best_candidate(class_hits, caller.start_line)
            else:
                module_hits = indexes.methods_by_python_module_and_name.get((module_name, callsite.callee_name), [])
                if module_hits:
                    return self.choose_best_candidate(module_hits, caller.start_line)
        return None

    @staticmethod
    def choose_best_candidate(candidates: List[MethodInfo], caller_line: int) -> MethodInfo:
        """Choose the nearest candidate by line distance."""
        return sorted(candidates, key=lambda item: abs(item.start_line - caller_line))[0]

    @staticmethod
    def extract_imports_for_file(metadata: Dict[str, object]) -> Dict[str, str]:
        """Extract imports dictionary from parser metadata."""
        imports = metadata.get("imports", {})
        if isinstance(imports, dict):
            return {str(key): str(value) for key, value in imports.items()}
        return {}

    @staticmethod
    def split_module_and_symbol(imported_ref: str) -> tuple[str, str | None]:
        """Split dotted import path into module + symbol."""
        tokens = imported_ref.split(".")
        if len(tokens) <= 1:
            return imported_ref, None
        return ".".join(tokens[:-1]), tokens[-1]

    @staticmethod
    def parse_method_id(method_id: str) -> tuple[str, str, str, int] | None:
        """Parse method_id formatted as file::class::method::line."""
        parts = method_id.rsplit("::", 3)
        if len(parts) != 4:
            return None
        file_path, class_name, method_name, start_line_text = parts
        class_value = "" if class_name == "<global>" else class_name
        try:
            start_line = int(start_line_text)
        except ValueError:
            return None
        return file_path, class_value, method_name, start_line

    @staticmethod
    def looks_like_source_file(relative_path: str) -> bool:
        """Heuristic filter for unknown-language source candidates."""
        path = relative_path.lower()
        if "/.git/" in path or path.endswith(".git"):
            return False
        file_name = Path(path).name
        if "." not in file_name:
            return False
        suffix = Path(file_name).suffix
        if not suffix:
            return False
        if len(suffix) > 8:
            return False
        binary_like_suffixes = {
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".bmp",
            ".ico",
            ".pdf",
            ".zip",
            ".gz",
            ".jar",
            ".class",
            ".dll",
            ".so",
            ".dylib",
            ".exe",
            ".bin",
            ".woff",
            ".woff2",
            ".ttf",
            ".otf",
            ".mp3",
            ".mp4",
            ".mov",
            ".avi",
            ".wasm",
        }
        return suffix not in binary_like_suffixes


