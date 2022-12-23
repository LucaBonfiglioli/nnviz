from __future__ import annotations

import typing as t
from uuid import uuid4

import networkx as nx
import torch.fx as fx
import torch.nn as nn
from torchvision.models import feature_extraction

from nnviz import entities as ent
from nnviz import inspection as insp


class ExtendedFxGraph(fx.graph.Graph):
    """Extended version of `torch.fx.Graph` that adds some useful properties. Like:
    - `qualnames`: Mapping from node to qualified name.
    - `callables`: Mapping from node to its respecitve script/function/nn.Module.
    - `edges`: List of edges in the graph.
    """

    def __init__(
        self,
        wrapped: fx.graph.Graph,
        qualnames: t.Mapping[fx.node.Node, str],
        callables: t.Dict[fx.node.Node, nn.Module | t.Callable],
    ) -> None:
        """Constructor.

        Args:
            wrapped (fx.graph.Graph): The wrapped graph.
            qualnames (t.Mapping[fx.node.Node, str]): A mapping from node to its qualified name.
            callables (t.Dict[fx.node.Node, nn.Module  |  t.Callable]): A mapping from node to its
                respective script/function/nn.Module.
        """
        super().__init__(
            wrapped.owning_module, wrapped._tracer_cls, wrapped._tracer_extras
        )
        self._wrapped = wrapped
        self._edges: t.Optional[t.Sequence[t.Tuple[t.Any, fx.node.Node]]] = None
        self._qualnames = qualnames
        self._callables = callables

    def _compute_edges(self) -> None:
        self._edges = []
        for node in self.nodes:
            for arg in set(node.args).union(set(node.kwargs.values())):
                self._edges.append((arg, node))

    @property
    def nodes(self):
        """Returns the nodes in the graph."""
        return self._wrapped.nodes

    @property
    def edges(self) -> t.Sequence[t.Tuple[t.Any, fx.node.Node]]:
        """Returns the edges in the graph as a list of tuples (arg, node)."""
        if self._edges is None:
            self._compute_edges()
        return self._edges  # type: ignore

    @property
    def qualnames(self) -> t.Mapping[fx.node.Node, str]:
        """Returns a mapping from node to its qualified name."""
        return self._qualnames

    @property
    def callables(self) -> t.Dict[fx.node.Node, nn.Module | t.Callable]:
        """Returns a mapping from node to its respective script/function/nn.Module."""
        return self._callables


class ExtendedNodePathTracer(feature_extraction.NodePathTracer):
    """Extended version of `torchvision.models.feature_extraction.NodePathTracer` that generates
    an `ExtendedFxGraph` instead of a `torch.fx.Graph`.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._qualname_to_callable: t.Dict[str, t.Callable] = {}

    def trace(
        self,
        root: nn.Module | t.Callable[..., t.Any],
        concrete_args: t.Optional[t.Dict[str, t.Any]] = None,
    ) -> ExtendedFxGraph:
        # Reset the qualname to callable mapping
        self._qualname_to_callable = {}

        # Trace the graph
        wrapped = super().trace(root, concrete_args)

        # Convert the callables (qualname -> callable) to a mapping (node -> callables)
        callables = {}
        # TODO: t.Callable is not a valid type for callables...
        default_callable = t.Callable
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

    def call_module(self, m: nn.Module, forward: t.Callable, args, kwargs):
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


class TorchFxInspector(insp.NNInspector):
    """NNInspector implementation with torch.fx."""

    def inspect(self, model: nn.Module) -> ent.NNGraph:
        tracer = ExtendedNodePathTracer()
        fxgraph = tracer.trace(model)
        return self._to_nngraph(fxgraph)

    def _to_nngraph(self, fxgraph: ExtendedFxGraph) -> ent.NNGraph:
        # Initialize a networkx graph
        nxgraph = nx.DiGraph()

        def opnode(node: fx.node.Node) -> ent.NodeModel:
            node_name = fxgraph.qualnames.get(node, node.name)
            node_path = node_name.split(".")
            callable_ = fxgraph.callables[node]
            if isinstance(callable_, nn.Module):
                full_op = str(callable_.__class__)
                op = callable_.__class__.__name__
            else:
                full_op = str(callable_)
                op = callable_.__name__
            return ent.OpNodeModel(
                name=node_name, path=node_path, op=op, full_op=full_op
            )

        # Populate graph
        for source, target in fxgraph.edges:
            source_name = (
                source.name if isinstance(source, fx.node.Node) else str(uuid4())
            )
            target_name = target.name

            # Create the edge
            nxgraph.add_edge(source_name, target_name)

            # Convert to node model both source and target
            target_model = opnode(target)

            if isinstance(source, fx.node.Node):
                source_model = opnode(source)
            else:
                # Note: use the target path as the source path if the source is not a node
                source_model = ent.ConstantNodeModel(
                    name=str(source),
                    path=target_model.path,
                    value=source,
                    value_type=source.__class__.__name__,
                )

            # Update the graph
            nxgraph.nodes[source_name].update({ent.NNGraph.MODEL_KEY: source_model})
            nxgraph.nodes[target_name].update({ent.NNGraph.MODEL_KEY: target_model})

        return ent.NNGraph(nxgraph)
