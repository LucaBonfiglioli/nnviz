from __future__ import annotations

from typing import Any, Dict, Iterable, Literal, Optional, Sequence, Tuple

import networkx as nx
from pydantic import BaseModel, Field


class NodeModel(BaseModel):
    type_: Literal[""] = ""

    name: str
    path: Sequence[str] = Field(default_factory=list)

    def depth(self) -> int:
        return len(self.path)


class OpNodeModel(NodeModel):
    type_: Literal["op"] = "op"

    op: str
    target: str

    symbol: Optional[str] = None
    full_symbol: Optional[str] = None


class ConstantNodeModel(NodeModel):
    type_: Literal["constant"] = "constant"

    value: Any
    value_type: str


class CollapsedNodeModel(NodeModel):
    type_: Literal["collapsed"] = "collapsed"


class NNGraph:
    MODEL_KEY = "model"

    @classmethod
    def empty(cls) -> NNGraph:
        return cls(nx.DiGraph())

    def __init__(self, graph: nx.DiGraph) -> None:
        self._graph = graph

    def __getitem__(self, name: str) -> NodeModel:
        return self._graph.nodes[name][self.MODEL_KEY]

    def __setitem__(self, name: str, model: NodeModel) -> None:
        if name in self._graph.nodes:
            self._graph.nodes[name][self.MODEL_KEY] = model
        else:
            self._graph.add_node(name, model=model)

    @property
    def nodes(self) -> Iterable[str]:
        yield from self._graph.nodes

    @property
    def edges(self) -> Iterable[Tuple[str, str]]:
        yield from self._graph.edges

    def add_edge(self, source: str, target: str) -> None:
        self._graph.add_edge(source, target)

    def collapse(self, depth: int) -> NNGraph:
        """Collapse the graph nodes that share the same path up to the given depth."""
        # Create a mapping (path -> node) for each node in the graph
        path_to_node = {}
        for node in self.nodes:
            node_model = self[node]
            path = tuple(node_model.path[:depth])
            path_to_node[path] = node

        # Create a mapping (node -> collapsed node)
        node_to_collapsed: Dict[str, Tuple[str, NodeModel]] = {}
        for node in self.nodes:
            node_model = self[node]
            if len(node_model.path) <= depth:
                node_to_collapsed[node] = node, node_model
            else:
                path = tuple(node_model.path[:depth])
                collapsed_node = path_to_node[path]
                collapsed_model = CollapsedNodeModel(name=collapsed_node, path=path)
                node_to_collapsed[node] = collapsed_node, collapsed_model

        # Create a new graph
        collapsed = self.empty()

        # Populate the new graph
        for source, target in self.edges:
            collapsed_src, model_src = node_to_collapsed[source]
            collapsed_tgt, model_tgt = node_to_collapsed[target]
            collapsed[collapsed_src] = model_src
            collapsed[collapsed_tgt] = model_tgt
            collapsed.add_edge(collapsed_src, collapsed_tgt)

        return collapsed
