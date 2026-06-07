"""Reusable indexes for call target resolution."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

from commit_feature_suite.models import MethodInfo


@dataclass
class GraphIndexes:
    """Precomputed method indexes for fast resolver lookup."""

    methods_by_name_and_file: Dict[Tuple[str, str], List[MethodInfo]]
    methods_by_name_and_class_and_file: Dict[Tuple[str, str, str], List[MethodInfo]]
    methods_by_name_global: Dict[str, List[MethodInfo]]
    methods_by_class_and_name_global: Dict[Tuple[str, str], List[MethodInfo]]
    methods_by_python_module_and_name: Dict[Tuple[str, str], List[MethodInfo]]
    methods_by_python_module_class_and_name: Dict[Tuple[str, str, str], List[MethodInfo]]


def python_module_name(file_path: str) -> str:
    """Convert Python file path to module name."""
    path = file_path.replace("\\", "/")
    if path.endswith(".py"):
        path = path[:-3]
    if path.endswith("/__init__"):
        path = path[: -len("/__init__")]
    return path.replace("/", ".")


def build_indexes(methods_by_id: Dict[str, MethodInfo]) -> GraphIndexes:
    """Build all indexes used by the resolver."""
    methods_by_name_and_file: Dict[Tuple[str, str], List[MethodInfo]] = defaultdict(list)
    methods_by_name_and_class_and_file: Dict[Tuple[str, str, str], List[MethodInfo]] = defaultdict(list)
    methods_by_name_global: Dict[str, List[MethodInfo]] = defaultdict(list)
    methods_by_class_and_name_global: Dict[Tuple[str, str], List[MethodInfo]] = defaultdict(list)
    methods_by_python_module_and_name: Dict[Tuple[str, str], List[MethodInfo]] = defaultdict(list)
    methods_by_python_module_class_and_name: Dict[Tuple[str, str, str], List[MethodInfo]] = defaultdict(list)

    for method in methods_by_id.values():
        methods_by_name_and_file[(method.file_path, method.method_name)].append(method)
        methods_by_name_and_class_and_file[(method.file_path, method.class_name, method.method_name)].append(method)
        methods_by_name_global[method.method_name].append(method)
        if method.class_name:
            methods_by_class_and_name_global[(method.class_name, method.method_name)].append(method)
        if method.language == "python":
            module_name = python_module_name(method.file_path)
            methods_by_python_module_and_name[(module_name, method.method_name)].append(method)
            if method.class_name:
                methods_by_python_module_class_and_name[
                    (module_name, method.class_name, method.method_name)
                ].append(method)

    return GraphIndexes(
        methods_by_name_and_file=dict(methods_by_name_and_file),
        methods_by_name_and_class_and_file=dict(methods_by_name_and_class_and_file),
        methods_by_name_global=dict(methods_by_name_global),
        methods_by_class_and_name_global=dict(methods_by_class_and_name_global),
        methods_by_python_module_and_name=dict(methods_by_python_module_and_name),
        methods_by_python_module_class_and_name=dict(methods_by_python_module_class_and_name),
    )


