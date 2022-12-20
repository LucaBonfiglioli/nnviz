from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from uuid import uuid4

import networkx as nx
import pygraphviz as pgv
import torch.nn as nn
from pydantic import BaseModel, Field
from torch.fx.graph import Graph as FxGraph
from torch.fx.node import Node as FxNode
from torchvision.models.feature_extraction import NodePathTracer
from hashlib import sha256


class ExtendedFxGraph(FxGraph):
    def __init__(
        self,
        wrapped: FxGraph,
        qualnames: Mapping[FxNode, str],
        callables: Dict[FxNode, Union[nn.Module, Callable]],
    ) -> None:
        super().__init__(
            wrapped.owning_module, wrapped._tracer_cls, wrapped._tracer_extras
        )
        self._wrapped = wrapped
        self._edges: Optional[Sequence[Tuple[Any, FxNode]]] = None
        self._qualnames = qualnames
        self._callables = callables

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

    @property
    def callables(self) -> Dict[FxNode, Union[nn.Module, Callable]]:
        return self._callables


class ExtendedNodePathTracer(NodePathTracer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._qualname_to_callable: Dict[str, Callable] = {}

    def trace(
        self,
        root: Union[nn.Module, Callable[..., Any]],
        concrete_args: Optional[Dict[str, Any]] = None,
    ) -> ExtendedFxGraph:
        # Reset the qualname to callable mapping
        self._qualname_to_callable = {}

        # Trace the graph
        wrapped = super().trace(root, concrete_args)

        # Convert the callables (qualname -> callable) to a mapping (node -> callables)
        callables = {}
        default_callable = Callable
        for node in wrapped.nodes:
            qualname = self.node_to_qualname.get(node, "__INVALID__")
            callables[node] = self._qualname_to_callable.get(qualname, node.target)
            if isinstance(callables[node], str):
                callables[node] = default_callable

        # Create the extended graph
        graph = ExtendedFxGraph(
            wrapped, qualnames=self.node_to_qualname, callables=callables
        )
        return graph

    def call_module(self, m: nn.Module, forward: Callable, args, kwargs):
        old_qualname = self.current_module_qualname
        try:
            module_qualname = self.path_of_module(m)
            self._qualname_to_callable[module_qualname] = m
            self.current_module_qualname = module_qualname
            if not self.is_leaf_module(m, module_qualname):
                out = forward(*args, **kwargs)
                return out
            return self.create_proxy("call_module", module_qualname, args, kwargs)
        finally:
            self.current_module_qualname = old_qualname


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


def to_nx(fxgraph: ExtendedFxGraph) -> nx.DiGraph:
    # Initialize a networkx graph
    nxgraph = nx.DiGraph()

    def opnode(node: FxNode) -> NodeModel:
        node_name = fxgraph.qualnames.get(node, node.name)
        node_path = node_name.split(".")
        target = node.target if isinstance(node.target, str) else node.target.__name__
        callable_ = fxgraph.callables[node]
        if isinstance(callable_, nn.Module):
            full_symbol = str(callable_.__class__)
            symbol = callable_.__class__.__name__
        else:
            full_symbol = str(callable_)
            symbol = callable_.__name__
        return OpNodeModel(
            name=node_name,
            path=node_path,
            op=node.op,
            target=target,
            symbol=symbol,
            full_symbol=full_symbol,
        )

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
                name=str(source),
                path=target_model.path,
                value=source,
                value_type=source.__class__.__name__,
            )

        # Update the graph
        nxgraph.nodes[source_name].update({"model": source_model})
        nxgraph.nodes[target_name].update({"model": target_model})

    return nxgraph


def collapse(nxgraph: nx.DiGraph, depth: int) -> nx.DiGraph:
    """Collapse the graph nodes that share the same path up to the given depth."""
    # Create a mapping (path -> node) for each node in the graph
    path_to_node = {}
    for node in nxgraph.nodes:
        node_model = nxgraph.nodes[node]["model"]
        path = tuple(node_model.path[:depth])
        path_to_node[path] = node

    # Create a mapping (node -> collapsed node)
    node_to_collapsed_models = {}
    node_to_collapsed_node = {}
    for node in nxgraph.nodes:
        node_model = nxgraph.nodes[node]["model"]
        if len(node_model.path) <= depth:
            node_to_collapsed_models[node] = node_model
            node_to_collapsed_node[node] = node
        else:
            path = tuple(node_model.path[:depth])
            collapsed_node = path_to_node[path]
            node_to_collapsed_models[node] = CollapsedNodeModel(
                name=collapsed_node, path=path
            )
            node_to_collapsed_node[node] = collapsed_node

    # Create a new graph
    collapsed = nx.DiGraph()

    # Populate the new graph
    for source, target in nxgraph.edges:
        collapsed.add_edge(
            node_to_collapsed_node[source], node_to_collapsed_node[target]
        )

    # Update the models
    for node in collapsed.nodes:
        collapsed.nodes[node].update(
            {"model": node_to_collapsed_models.get(node, nxgraph.nodes[node]["model"])}
        )

    return collapsed


class GraphDrawer:
    def __init__(self) -> None:
        self._default_node_params = {
            "fontname": "Arial",
            "shape": "box",
            "style": "rounded,filled",
            "margin": "0.2,0.1",
            "color": "gray",
            "fillcolor": "gray",
        }
        self._default_edge_params = {
            "fontname": "Arial",
            "color": "black",
            "penwidth": "2.0",
        }

    def _get_color(self, value) -> str:
        hash_ = sha256(str(value).encode("utf-8")).hexdigest()
        index = int(hash_, 16) % (2**24)
        r = (index >> 16) & 0xFF
        g = (index >> 8) & 0xFF
        b = index & 0xFF
        return f"#{r:02x}{g:02x}{b:02x}"

    def _text_color(self, color: str) -> str:
        r, g, b = tuple(int(color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
        return "black" if (r * 0.299 + g * 0.587 + b * 0.114) > 186 else "white"

    def _constant_params(self, node: ConstantNodeModel) -> Dict[str, Any]:
        if isinstance(node.value, (int, float)):
            value = f"{node.value:.2f}"
        else:
            value = str(node.value)

        color = self._get_color(node.value_type)

        return {
            "label": f'<<B><FONT POINT-SIZE="20">{value}</FONT></B> <BR/><BR/> <I>{node.value_type}</I>>',
            "color": color,
            "fillcolor": color,
            "fontcolor": self._text_color(color),
        }

    def _op_params(self, node: OpNodeModel) -> Dict[str, Any]:
        color = self._get_color(node.full_symbol)
        return {
            "label": f'<<B><FONT POINT-SIZE="20">{node.symbol}</FONT></B> <BR/><BR/> <I>{node.name}</I>>',
            "color": color,
            "fillcolor": color,
            "fontcolor": self._text_color(color),
        }

    def _collapsed_params(self, node: CollapsedNodeModel) -> Dict[str, Any]:
        color = self._get_color(node.path[:-1])
        return {
            "label": f'<<B><FONT POINT-SIZE="20">{".".join(node.path)}</FONT></B>>',
            "color": color,
            "fillcolor": color,
            "fontcolor": self._text_color(color),
        }

    def _node_params(self, node: NodeModel) -> Dict[str, Any]:
        type_map = {
            "constant": self._constant_params,
            "op": self._op_params,
            "collapsed": self._collapsed_params,
        }
        params = type_map[node.type_](node)
        return {**self._default_node_params, **params}

    def draw(self, nxgraph: nx.DiGraph) -> pgv.AGraph:
        # Initialize a pygraphviz graph
        pgvgraph = pgv.AGraph(directed=True, strict=True)

        # Populate nodes
        for node in nxgraph.nodes:
            model: NodeModel = nxgraph.nodes[node]["model"]
            pgvgraph.add_node(node, **self._node_params(model))

        # Populate edges
        for source, target in nxgraph.edges:
            pgvgraph.add_edge(source, target, **self._default_edge_params)

        return pgvgraph

    def show(self, graph: pgv.AGraph, path: str) -> None:
        graph.draw(path, prog="dot")


if __name__ == "__main__":

    # Mock model for testing
    from torchvision.models import efficientnet_b0

    to_insepct = efficientnet_b0()

    # Create a tracer
    kwargs = {}  # TODO: Add default kwargs
    tracer = ExtendedNodePathTracer(**kwargs)

    # Trace the model
    nn_graph = tracer.trace(to_insepct)

    # Convert to networkx graph
    G = to_nx(nn_graph)

    # Collapse the graph
    import sys

    G = collapse(G, depth=int(sys.argv[1]))

    # Draw the graph
    drawer = GraphDrawer()
    A = drawer.draw(G)
    drawer.show(A, f"test{sys.argv[1]}.pdf")
