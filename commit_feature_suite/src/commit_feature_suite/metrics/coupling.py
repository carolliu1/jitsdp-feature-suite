"""Method coupling metric collection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class MethodCouplingRecord:
    """Coupling metrics for one method node in call graph."""

    in_coupling: int
    out_coupling: int


class MethodCouplingCollector:
    """Collect coupling metrics from graph for affected methods."""

    @staticmethod
    def build_method_coupling_map(*, graph_result, affected_methods) -> Dict[str, MethodCouplingRecord]:
        graph = graph_result.graph
        method_ids = {item.method.method_id for item in affected_methods}
        result: Dict[str, MethodCouplingRecord] = {}
        for method_id in method_ids:
            result[method_id] = MethodCouplingRecord(
                in_coupling=int(graph.in_degree(method_id)),
                out_coupling=int(graph.out_degree(method_id)),
            )
        return result

    @staticmethod
    def degree_values(*, method_coupling_map: Dict[str, MethodCouplingRecord]) -> tuple[List[int], List[int]]:
        in_values = [record.in_coupling for record in method_coupling_map.values()]
        out_values = [record.out_coupling for record in method_coupling_map.values()]
        return in_values, out_values

    @staticmethod
    def get_for_method(*, method_id: str, method_coupling_map: Dict[str, MethodCouplingRecord]) -> MethodCouplingRecord | None:
        return method_coupling_map.get(method_id)

