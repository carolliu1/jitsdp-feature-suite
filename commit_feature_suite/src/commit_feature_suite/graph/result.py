"""Graph build result models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import networkx as nx

from commit_feature_suite.models import MethodInfo


@dataclass
class CallGraphBuildResult:
    """Holds the graph plus method lookup tables."""

    graph: nx.DiGraph
    methods_by_id: Dict[str, MethodInfo]
    methods_by_file: Dict[str, List[MethodInfo]]
    node_count: int
    file_count: int


