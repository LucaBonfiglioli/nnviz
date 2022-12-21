from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union
from uuid import uuid4

import networkx as nx
import torch.nn as nn
from torch.fx.graph import Graph as FxGraph
from torch.fx.node import Node as FxNode
from torchvision.models.feature_extraction import NodePathTracer
from nnviz.entities import NodeModel, OpNodeModel, ConstantNodeModel, NNGraph
from nnviz.inspection.base import NNInspector


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
        return self._edges  # type: ignore

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


class TorchFxInspector(NNInspector):
    def inspect(self, model: nn.Module) -> NNGraph:
        tracer = ExtendedNodePathTracer()
        fxgraph = tracer.trace(model)
        return self._to_nngraph(fxgraph)

    def _to_nngraph(self, fxgraph: ExtendedFxGraph) -> NNGraph:
        # Initialize a networkx graph
        nxgraph = nx.DiGraph()

        def opnode(node: FxNode) -> NodeModel:
            node_name = fxgraph.qualnames.get(node, node.name)
            node_path = node_name.split(".")
            target = (
                node.target if isinstance(node.target, str) else node.target.__name__
            )
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

        return NNGraph(nxgraph)
