from __future__ import annotations

import typing as t
from functools import cached_property
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
        self._qualnames = qualnames
        self._callables = callables

    @property
    def nodes(self):
        """Returns the nodes in the graph."""
        return self._wrapped.nodes

    @cached_property
    def edges(self) -> t.Sequence[t.Tuple[fx.node.Node, fx.node.Node]]:
        """Returns the edges in the graph as a list of tuples (arg, node)."""
        edges = []
        for node in self.nodes:
            for arg in [*node.args] + [*node.kwargs.values()]:
                if isinstance(arg, fx.node.Node):
                    edges.append((arg, node))
        return edges

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
        for node in wrapped.nodes:
            qualname = self.node_to_qualname.get(node, "__INVALID__")
            callables[node] = self._qualname_to_callable.get(qualname, node.target)

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

    def _convert_node(
        self, fxgraph: ExtendedFxGraph, node: fx.node.Node
    ) -> ent.NodeModel:
        node_map = {
            "call_module": self._op_node,
            "call_function": self._op_node,
            "call_method": self._op_node,
            "placeholder": self._input_node,
            "output": self._output_node,
            "get_attr": self._op_node,  # TODO: handle get_attr
        }

        return node_map[node.op](fxgraph, node)

    def _input_node(
        self, fxgraph: ExtendedFxGraph, node: fx.node.Node
    ) -> ent.NodeModel:
        node_name = fxgraph.qualnames.get(node, node.name)
        node_path = node_name.split(".")
        return ent.InputNodeModel(name=node_name, path=node_path)

    def _output_node(
        self, fxgraph: ExtendedFxGraph, node: fx.node.Node
    ) -> ent.NodeModel:
        node_name = fxgraph.qualnames.get(node, node.name)
        node_path = node_name.split(".")
        return ent.OutputNodeModel(name=node_name, path=node_path)

    def _extract_args(self, node: fx.node.Node) -> t.List[t.Any]:
        # Extract the constant arguments
        const_args = []
        for arg in node.args:
            if isinstance(arg, fx.node.Node):
                continue
            const_args.append(arg)

        return const_args

    def _extract_kwargs(self, node: fx.node.Node) -> t.Dict[str, t.Any]:
        # Extract the constant arguments
        const_kwargs = {}
        for k, v in node.kwargs.items():
            if isinstance(v, fx.node.Node):
                continue
            const_kwargs[k] = v

        return const_kwargs

    def _op_node(self, fxgraph: ExtendedFxGraph, node: fx.node.Node) -> ent.NodeModel:
        node_name = fxgraph.qualnames.get(node, node.name)
        node_path = node_name.split(".")
        callable_ = fxgraph.callables.get(node, node.target)
        if isinstance(callable_, nn.Module):
            op = callable_.__class__.__name__
            full_op = ".".join([str(callable_.__module__), op])
        elif isinstance(callable_, t.Callable):
            op = callable_.__name__
            full_op = ".".join([str(callable_.__module__), op])
        else:
            full_op = str(callable_)
            op = str(callable_)

        # Create the node model
        return ent.OpNodeModel(
            name=node_name,
            path=node_path,
            op=op,
            full_op=full_op,
            const_args=self._extract_args(node),
            const_kwargs=self._extract_kwargs(node),
        )

    def _to_nngraph(self, fxgraph: ExtendedFxGraph) -> ent.NNGraph:
        # Initialize a networkx graph
        nxgraph = nx.DiGraph()
        # Populate graph
        for src, tgt in fxgraph.edges:
            src_name = src.name if isinstance(src, fx.node.Node) else str(uuid4())
            tgt_name = tgt.name

            # Create the edge
            nxgraph.add_edge(src_name, tgt_name)

            # Convert to node model both source and target
            src_model = self._convert_node(fxgraph, src)
            tgt_model = self._convert_node(fxgraph, tgt)

            # Update the graph
            nxgraph.nodes[src_name].update({ent.NNGraph.MODEL_KEY: src_model})
            nxgraph.nodes[tgt_name].update({ent.NNGraph.MODEL_KEY: tgt_model})

        return ent.NNGraph(nxgraph)
