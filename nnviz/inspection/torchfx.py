from __future__ import annotations

import functools
import typing as t
from uuid import uuid4

import networkx as nx
import torch
import torch.fx as fx
import torch.nn as nn
from torchvision.models import feature_extraction

from nnviz import dataspec as ds
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
        specs: t.Optional[t.Mapping[fx.node.Node, ds.DataSpec]] = None,
    ) -> None:
        """Constructor.

        Args:
            wrapped (fx.graph.Graph): The wrapped graph.
            qualnames (t.Mapping[fx.node.Node, str]): A mapping from node to its qualified name.
            callables (t.Dict[fx.node.Node, nn.Module  |  t.Callable]): A mapping from node to its
                respective script/function/nn.Module.
            specs (t.Optional[t.Mapping[fx.node.Node, ds.DataSpec]], optional):
                A mapping from node to its output specs. Defaults to None.
        """
        super().__init__(
            wrapped.owning_module, wrapped._tracer_cls, wrapped._tracer_extras
        )
        self._wrapped = wrapped
        self._qualnames = qualnames
        self._callables = callables
        self._specs = specs or {}

    @property
    def nodes(self):
        """Returns the nodes in the graph."""
        return self._wrapped.nodes

    def _recurse_args(self, args: t.Iterable[t.Any]) -> t.List[fx.node.Node]:
        deps = []
        for arg in args:
            if isinstance(arg, fx.node.Node):
                deps.append(arg)
                continue

            if isinstance(arg, t.Sequence) and not isinstance(arg, str):
                deps += self._recurse_args(arg)
            elif isinstance(arg, t.Mapping):  # pragma: no cover
                deps += self._recurse_args(arg.values())

        return deps

    def _get_dependencies(self, node: fx.node.Node) -> t.List[fx.node.Node]:
        return self._recurse_args([*node.args] + [*node.kwargs.values()])

    @functools.cached_property
    def edges(self) -> t.Sequence[t.Tuple[fx.node.Node, fx.node.Node]]:
        """Returns the edges in the graph as a list of tuples (arg, node)."""
        edges = []
        for node in self.nodes:
            for arg in self._get_dependencies(node):
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

    @property
    def specs(self) -> t.Mapping[fx.node.Node, ds.DataSpec]:
        """Returns a mapping from node to its output data specs."""
        return self._specs


class ExtendedNodePathTracer(feature_extraction.NodePathTracer):
    """Extended version of `torchvision.models.feature_extraction.NodePathTracer` that generates
    an `ExtendedFxGraph` instead of a `torch.fx.Graph`.
    """

    def __init__(self, *args, inputs: t.Optional[t.Mapping[str, t.Any]] = None) -> None:
        super().__init__(*args)
        self._qualname_to_callable: t.Dict[str, t.Callable] = {}
        self._inputs = inputs

    def trace(
        self,
        root: nn.Module | t.Callable[..., t.Any],
        concrete_args: t.Optional[t.Dict[str, t.Any]] = None,
    ) -> ExtendedFxGraph:
        # Reset the qualname to callable mapping
        self._qualname_to_callable = {}

        # Trace the graph
        wrapped = super().trace(root, concrete_args)
        gm = fx.graph_module.GraphModule(root, wrapped)  # type: ignore

        # Convert the callables (qualname -> callable) to a mapping (node -> callables)
        callables = {}
        for node in wrapped.nodes:
            qualname = self.node_to_qualname.get(node, "__INVALID__")
            callables[node] = self._qualname_to_callable.get(qualname, node.target)

        specs = self._build_specs(gm)

        # Create the extended graph
        graph = ExtendedFxGraph(
            wrapped, qualnames=self.node_to_qualname, callables=callables, specs=specs
        )
        return graph

    def _build_specs(
        self, root: nn.Module | t.Callable[..., t.Any]
    ) -> t.Optional[t.Mapping[fx.node.Node, ds.DataSpec]]:
        # Abort if no inputs are provided
        if self._inputs is None or not isinstance(root, nn.Module):
            return None

        # Collect all callables and their respective nodes
        nodes = []
        callables = []
        input_nodes: t.List[fx.node.Node] = []
        for node, qualname in self.node_to_qualname.items():
            if hasattr(node, "op") and node.op == "placeholder":
                input_nodes.append(node)
            if qualname in self._qualname_to_callable:
                nodes.append(node)
                callables.append(self._qualname_to_callable[qualname])

        specs = {
            node: ds.DataSpec.build(self._inputs[node.name]) for node in input_nodes
        }

        def register_specs(the_self: nn.Module, input: t.Any, output: t.Any):
            node = nodes[callables.index(the_self)]
            output_specs = ds.DataSpec.build(output)
            specs[node] = output_specs

        def get_children(model: torch.nn.Module) -> t.List[torch.nn.Module]:
            children = list(model.children())
            if len(children) == 0:
                return [model]
            else:
                recur_children = []
                for child in children:
                    recur_children.extend(get_children(child))
            return recur_children

        # Register hooks
        recursive_children: t.List[nn.Module] = get_children(root)
        for module in recursive_children:
            if module in callables:
                module.register_forward_hook(register_specs)

        # Run the model
        with torch.no_grad():
            root(**self._inputs)

        return specs

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

    def inspect(
        self, model: nn.Module, inputs: t.Optional[t.Mapping[str, t.Any]] = None
    ) -> ent.NNGraph:
        tracer = ExtendedNodePathTracer(inputs=inputs)
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
        nngraph = ent.NNGraph(nxgraph)

        # Populate graph
        for src, tgt in fxgraph.edges:
            src_name = src.name if isinstance(src, fx.node.Node) else str(uuid4())
            tgt_name = tgt.name

            # Create the edge
            spec = fxgraph.specs.get(src, None)
            nngraph.add_edge(src_name, tgt_name, spec=spec)

            # Convert to node model both source and target
            src_model = self._convert_node(fxgraph, src)
            tgt_model = self._convert_node(fxgraph, tgt)

            # Update the graph
            nngraph[src_name] = src_model
            nngraph[tgt_name] = tgt_model

        return nngraph
