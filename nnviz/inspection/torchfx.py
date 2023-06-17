from __future__ import annotations

import functools
import importlib
import typing as t
import warnings
from uuid import uuid4

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
            qualnames (t.Mapping[fx.node.Node, str]): A mapping from node to its
                qualified name.
            callables (t.Dict[fx.node.Node, nn.Module  |  t.Callable]): A mapping from
                node to its respective script/function/nn.Module.
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

    @property
    def total_parameters(self) -> int:
        """Returns the total number of parameters in the graph."""
        acc = 0
        for callable in self.callables.values():
            if isinstance(callable, nn.Module):
                acc += sum(p.numel() for p in callable.parameters())
        return acc

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
    """Extended version of `torchvision.models.feature_extraction.NodePathTracer` that
    generates an `ExtendedFxGraph` instead of a `torch.fx.Graph`.
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
        try:
            wrapped = super().trace(root, concrete_args)
        except Exception as e:
            msg = (
                "You provided a model that cannot be traced with torch.fx and you may "
                "need to apply some changes to your source code. See the following "
                "link for more information: "
                "https://pytorch.org/docs/stable/fx.html#tracing\n\n"
                "Also keep in mind that dynamic control flow is currently not "
                "supported by torch.fx (e.g. if statements, while loops, etc.) and "
                "that you may need to manually convert your model to a static graph "
                "before tracing it. The easiest way to trace dynamic models is using "
                "the torchscript compiler, which is currently not supported by nnviz. "
                "If you really need to plot a dynamic model and you have some good "
                "ideas on how to implement it, please open an issue on GitHub and we "
                "will discuss it. Be warned that this is not an easy task and that it "
                "may require a lot of work."
            )
            raise RuntimeError(msg) from e
        gm = fx.graph_module.GraphModule(root, wrapped)  # type: ignore

        # Something is very wrong with the original torch.fx tracer. I am fixing it here
        for k, v in self.node_to_qualname.items():
            k: fx.node.Node
            v: str

            # Ignore if the node is already in the mapping
            if v in self._qualname_to_callable:
                continue

            # Ignore if the node is either a get_attr or a placeholder
            if k.op in ["placeholder", "get_attr"]:
                continue

            # If the target is a string (as it should) but no callable is found for it,
            # Then it probably means that the target is a reference to another node.
            # Example: when a layer is used multiple times in a model, every additional
            # time it is used, a new node is created that references the original node.
            # These nodes differ by their name (a _1, _2, etc... is appended) but they
            # point to the same target.
            # I am tired of this fuck fiesta so I am just going to kill it with fire.
            if (
                isinstance(k.target, str)
                and v not in self._qualname_to_callable
                and k.target in self._qualname_to_callable
            ):
                self._qualname_to_callable[v] = self._qualname_to_callable[k.target]

            # If the target is a callable, then we can just add it to the mapping. This
            # is the case when the node is a builtin function (e.g. torch.add)
            # How does it even work? I don't know. I don't care.
            elif callable(k.target):
                self._qualname_to_callable[v] = k.target

        # Convert the callables (qualname -> callable) to a mapping (node -> callables)
        callables = {}
        for node in wrapped.nodes:
            qualname = self.node_to_qualname.get(node, "__INVALID__")
            callables[node] = self._qualname_to_callable.setdefault(
                qualname, node.target
            )

        specs = self._build_specs(gm)

        # Create the extended graph
        graph = ExtendedFxGraph(
            wrapped, qualnames=self.node_to_qualname, callables=callables, specs=specs
        )
        return graph

    def _register_callbacks_to_callables(
        self,
        node_clb_pairs: t.List[t.Tuple[fx.node.Node, t.Any]],
        specs: t.Dict[fx.node.Node, ds.DataSpec],
    ) -> None:
        # For the same reason as in `trace` we need to address the case where a Module
        # is used multiple times in the graph. In this case, we need to register the
        # specs for all nodes that point to the same nn.Module.
        # Trust me, I don't like this either.
        def register_specs(the_self: nn.Module, input: t.Any, output: t.Any):
            tgt_nodes = [x for x, y in node_clb_pairs if y is the_self]
            specs.update({node: ds.DataSpec.build(output) for node in tgt_nodes})

        # It must be done for pure functions as well...
        def build_fn_wrapper(node: fx.node.Node, clb: t.Any):
            def wrapper(*args, **kwargs):
                output = clb(*args, **kwargs)
                specs[node] = ds.DataSpec.build(output)
                return output

            return wrapper

        # ... and for methods. Goddammit.
        def build_method_wrapper(node: fx.node.Node, target: str):
            def wrapper(*args, **kwargs):
                method = getattr(args[0], target)
                output = method(*args[1:], **kwargs)
                specs[node] = ds.DataSpec.build(output)
                return output

            return wrapper

        # Register the hooks
        for node, clb in node_clb_pairs:
            if isinstance(clb, nn.Module):
                clb.register_forward_hook(register_specs)
            elif callable(clb):
                node.target = build_fn_wrapper(node, clb)
            elif node.op == "call_method":
                node.op = "call_function"
                node.target = build_method_wrapper(node, node.target)  # type: ignore

    def _build_specs(
        self, graph_module: fx.graph_module.GraphModule
    ) -> t.Optional[t.Mapping[fx.node.Node, ds.DataSpec]]:
        inputs = self._inputs

        # Abort if no inputs are provided
        if inputs is None:
            return None

        # Spec collection
        specs: t.Dict[fx.node.Node, ds.DataSpec] = {}

        # Special cases
        special_cases = {
            "placeholder": lambda x: ds.DataSpec.build(inputs.get(x.target)),
            "get_attr": lambda x: ds.DataSpec.build(
                graph_module.get_parameter(x.target)
            ),
        }

        # Collect all callables and their respective nodes
        node_clb_pairs: t.List[t.Tuple[fx.node.Node, t.Any]] = []
        for node, qualname in self.node_to_qualname.items():
            # Handle special cases first
            if node.op in special_cases:
                specs[node] = special_cases[node.op](node)

            # Register the node as a callable
            node_clb_pairs.append((node, self._qualname_to_callable[qualname]))

        # Register the hooks
        self._register_callbacks_to_callables(node_clb_pairs, specs)

        # This is the reason why people drink. To forget the horrors concealed in the
        # depths of torch.fx.
        # The open source community is not going to miss me. And I don't blame them.
        # I need a therapist.
        graph_module = fx.graph_module.GraphModule(graph_module, graph_module.graph)

        # Run the model
        try:
            with torch.no_grad():
                graph_module(**inputs)
        except Exception as e:
            input_pretty = ds.DataSpec.build(inputs).pretty()
            msg = (
                "Whooops! Something went wrong while performing the forward pass! \n"
                "You passed the following inputs: \n"
                f"\n{input_pretty}\n"
                "And you got the following error: \n"
                f"\n    '{e}'\n\n"
                "Things to check: \n"
                "  - Are the input names matching the 'forward()' signature? \n"
                "  - Are the input shapes and dtypes correct? \n\n"
                "The inspection will continue, just without the data specs tracing."
                "You can use the generated graph to help you debug the issue, that's "
                "why I made nnviz for!\n"
            )
            warnings.warn(msg)
            return None

        return specs

    def call_module(self, m: nn.Module, forward: t.Callable, args, kwargs):
        """Override the call_module method to keep track of the current module qualname.

        I have to override this docstring because the original one sucks so much that
        it cannot even be rendered by Sphinx. I'm not even kidding. For the brave, refer
        to the original pytorch source code, I have had enough of this.
        """
        old_qualname = self.current_module_qualname
        try:
            module_qualname = self.path_of_module(m)
            self._qualname_to_callable[module_qualname] = m
            self.current_module_qualname = module_qualname
            if not self.is_leaf_module(m, module_qualname):
                return forward(*args, **kwargs)
            return self.create_proxy("call_module", module_qualname, args, kwargs)
        finally:
            self.current_module_qualname = old_qualname


class TorchFxInspector(insp.NNInspector):
    """NNInspector implementation with torch.fx."""

    def inspect(
        self, model: nn.Module, inputs: t.Optional[t.Mapping[str, t.Any]] = None
    ) -> ent.NNGraph:
        # Initialize the tracer
        tracer = ExtendedNodePathTracer(inputs=inputs)

        # Trace the model
        fxgraph = tracer.trace(model)

        # Build the graph
        nngraph = self._to_nngraph(fxgraph)
        nngraph.metadata = self._build_metadata(model)

        return nngraph

    def _build_metadata(self, model: nn.Module) -> ent.GraphMeta:
        metadata = ent.GraphMeta()  # type: ignore
        metadata.title = model.__class__.__name__
        metadata.source = model.__class__.__module__

        try:
            root_model = model.__class__.__module__.split(".")[0]
            version = metadata.source_version = importlib.import_module(
                root_model
            ).__version__
            metadata.source_version = version

        # Attempt to get the version, but just don't do anything if it fails
        # Most likely the module is not installed or dynamically imported/generated
        # and we don't want to break the graph generation just for that.
        except Exception:  # pragma: no cover
            pass

        return metadata

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
        callable_ = fxgraph.callables.get(node, node.target)
        node_path = node_name.split(".")
        n_params, perc_params = 0, 0.0
        if isinstance(callable_, nn.Module):
            op = callable_.__class__.__name__
            full_op = ".".join([str(callable_.__module__), op])
            n_params = sum(p.numel() for p in callable_.parameters())
            if fxgraph.total_parameters > 0:  # pragma: no branch
                perc_params = n_params / fxgraph.total_parameters
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
            n_parameters=n_params,
            perc_parameters=perc_params,
        )

    def _to_nngraph(self, fxgraph: ExtendedFxGraph) -> ent.NNGraph:
        # Initialize a networkx graph
        nngraph = ent.NNGraph.empty()

        # Populate graph
        for src, tgt in fxgraph.edges:
            src_name = src.name if isinstance(src, fx.node.Node) else str(uuid4())
            tgt_name = tgt.name

            # Create the edge
            spec = fxgraph.specs.get(src, None)
            nngraph.add_edge(src_name, tgt_name, src.name, spec=spec)

            # Convert to node model both source and target
            src_model = self._convert_node(fxgraph, src)
            tgt_model = self._convert_node(fxgraph, tgt)

            # Update the graph
            nngraph[src_name] = src_model
            nngraph[tgt_name] = tgt_model

        return nngraph
