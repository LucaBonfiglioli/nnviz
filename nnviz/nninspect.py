from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union
from uuid import uuid4

import networkx as nx
import pygraphviz as pgv
import torch
from pydantic import BaseModel, Field, root_validator
from torch.fx.graph import Graph as FxGraph
from torch.fx.node import Node as FxNode
from torchvision.models.feature_extraction import NodePathTracer


class FxGraphWithEdges(FxGraph):
    def __init__(
        self, wrapped: FxGraph, qualnames: Optional[Mapping[FxNode, str]] = None
    ) -> None:
        super().__init__(
            wrapped.owning_module, wrapped._tracer_cls, wrapped._tracer_extras
        )
        self._wrapped = wrapped
        self._edges: Optional[Sequence[Tuple[str, str]]] = None
        self._qualnames = qualnames or {node: node.name for node in self.nodes}

    def _compute_edges(self) -> None:
        self._edges = []
        for node in self.nodes:
            for arg in set(node.args).union(set(node.kwargs.values())):
                self._edges.append((arg, node))

    @property
    def nodes(self):
        return self._wrapped.nodes

    @property
    def edges(self) -> Sequence[Tuple[Any, FxNode]]:
        if self._edges is None:
            self._compute_edges()
        return self._edges

    @property
    def qualnames(self) -> Mapping[FxNode, str]:
        return self._qualnames


class NodePathTracerWithEdges(NodePathTracer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._edges = {}

    def trace(
        self,
        root: Union[torch.nn.Module, Callable[..., Any]],
        concrete_args: Optional[Dict[str, Any]] = None,
    ) -> FxGraphWithEdges:
        wrapped = super().trace(root, concrete_args)
        graph = FxGraphWithEdges(wrapped, qualnames=self.node_to_qualname)
        return graph


class NodeModel(BaseModel):
    name: str
    path: Sequence[str] = Field(default_factory=list)

    def depth(self) -> int:
        return len(self.path)


class OpNodeModel(NodeModel):
    op: str
    target: str


class ConstantNodeModel(NodeModel):
    value: Any


def to_nx(fxgraph: FxGraphWithEdges) -> nx.DiGraph:
    # Initialize a networkx graph
    nxgraph = nx.DiGraph()

    def opnode(node: FxNode) -> NodeModel:
        node_name = fxgraph.qualnames.get(node, node.name)
        node_path = node.name.split(".")[:-1]
        target = node.target if isinstance(node.target, str) else node.target.__name__
        return OpNodeModel(name=node_name, path=node_path, op=node.op, target=target)

    # Populate graph
    for source, target in fxgraph.edges:
        source_name = source.name if isinstance(source, FxNode) else str(uuid4())
        target_name = target.name

        # Create the edge
        nxgraph.add_edge(source_name, target_name)

        # Convert to node model both source and target
        target_model = opnode(target)

        if isinstance(source, FxNode):
            source_model = opnode(source)
        else:
            # Note: use the target path as the source path if the source is not a node
            source_model = ConstantNodeModel(
                name=str(source), path=target_model.path, value=source
            )

        # Update the graph
        nxgraph.nodes[source_name].update({"model": source_model})
        nxgraph.nodes[target_name].update({"model": target_model})

    return nxgraph


def draw(nxgraph: nx.DiGraph) -> pgv.AGraph:
    # Initialize a pygraphviz graph
    pgvgraph = pgv.AGraph(directed=True, strict=True)

    ## TODO

    return pgvgraph


def show(graph: pgv.AGraph) -> None:
    graph.draw("test.pdf", prog="dot")


if __name__ == "__main__":

    # Mock model for testing
    from torchvision.models import efficientnet_b0

    to_insepct = efficientnet_b0()

    # Create a tracer
    kwargs = {}  # TODO: Add default kwargs
    tracer = NodePathTracerWithEdges(**kwargs)

    # Trace the model
    nn_graph = tracer.trace(to_insepct)

    # Convert to networkx graph
    G = to_nx(nn_graph)

    # Draw the graph
    A = draw(G)
    show(A)
