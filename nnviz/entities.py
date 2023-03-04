from __future__ import annotations

import typing as t

import networkx as nx
import pydantic as pyd

import nnviz
from nnviz import dataspec


class NodeModel(pyd.BaseModel):
    """Pydantic model for a node in the graph. Contains some information about a layer
    of a neural network."""

    type_: t.Literal[""] = ""

    name: str = pyd.Field(..., description="Qualified name of the node.")
    path: t.Sequence[str] = pyd.Field(
        default_factory=list,
        description="Path of the node. E.g. ['features', '0', 'conv1']",
    )

    def depth(self) -> int:
        """Returns the depth of the node in the graph."""
        return len(self.path)


class OpNodeModel(NodeModel):
    """`NodeModel` specialized for an operation node."""

    type_: t.Literal["op"] = "op"

    op: str = pyd.Field("", description="Name of the operation. E.g. 'Conv2d'.")
    full_op: str = pyd.Field(
        "",
        description="Full name of the operation. E.g. 'torch.nn.modules.conv.Conv2d'.",
    )
    const_args: t.Sequence[t.Any] = pyd.Field(
        default_factory=list,
        description="Constant positional arguments of the operation.",
    )
    const_kwargs: t.Mapping[str, t.Any] = pyd.Field(
        default_factory=dict, description="Constant keyword arguments of the operation."
    )
    n_parameters: int = pyd.Field(0, description="Number of parameters.")
    perc_parameters: float = pyd.Field(0.0, description="Percentage of parameters.")


class CollapsedNodeModel(NodeModel):
    """`NodeModel` specialized for a collapsed node."""

    type_: t.Literal["collapsed"] = "collapsed"

    n_parameters: int = pyd.Field(0, description="Number of parameters.")
    perc_parameters: float = pyd.Field(0.0, description="Percentage of parameters.")


class InputNodeModel(NodeModel):
    """`NodeModel` specialized for an input node."""

    type_: t.Literal["input"] = "input"


class OutputNodeModel(NodeModel):
    """`NodeModel` specialized for an output node."""

    type_: t.Literal["output"] = "output"


t_any_node = t.Union[
    NodeModel, OpNodeModel, CollapsedNodeModel, InputNodeModel, OutputNodeModel
]


class GraphMeta(pyd.BaseModel):
    """Pydantic model for the metadata associated with the graph."""

    title: str = pyd.Field("", description="Title of the graph.")
    description: str = pyd.Field("", description="General description of the graph.")
    source: str = pyd.Field("", description="Where the graph comes from.")
    nnviz_version: str = pyd.Field(
        nnviz.__version__, description="Version of nnviz used to generate the graph."
    )


class GraphData(pyd.BaseModel):
    """Pydantic model for the data associated with the graph. It consists of an anemic
    representation of the graph, i.e. a mapping of nodes and a list of edges.
    This is done to make it easier to serialize and deserialize the graph, without
    having to deal with the networkx graph representation.
    """

    nodes: t.Mapping[str, t_any_node] = pyd.Field(
        default_factory=dict, description="Mapping (node -> node model)."
    )
    edges: t.Sequence[t.Tuple[str, str, t.Optional[dataspec.t_any_spec]]] = pyd.Field(
        default_factory=list, description="List of edges. As tuples (src, tgt, [spec])."
    )
    metadata: GraphMeta = pyd.Field(
        GraphMeta(), description="Metadata associated with the graph."  # type: ignore
    )


class NNGraph:
    """Graph representation of a neural network."""

    MODEL_KEY = "model"
    """Key used to store the `NodeModel` in the graph nodes."""
    SPEC_KEY = "spec"
    """Key used to store the `DataSpec` in the graph edges."""

    @classmethod
    def empty(cls) -> NNGraph:
        """Create an empty graph."""
        return cls(nx.DiGraph(), GraphMeta.parse_obj({}))

    @classmethod
    def from_data(cls, data: GraphData) -> NNGraph:
        """Create a graph from the data associated with it.

        Args:
            data (GraphData): Data to create the graph from.

        Returns:
            NNGraph: The graph with fully populated nodes and edges.
        """
        graph = cls(nx.DiGraph(), data.metadata)
        for name, model in data.nodes.items():
            graph[name] = model
        for source, target, spec in data.edges:
            graph.add_edge(source, target, spec=spec)

        return graph

    def __init__(self, graph: nx.DiGraph, metadata: GraphMeta) -> None:
        """Constructor.

        Args:
            graph (nx.DiGraph): NetworkX graph wrapped by this class.
            metadata (GraphMeta): Metadata associated with the graph.
        """
        self._graph = graph
        self._metadata = metadata

    def __getitem__(self, name: str) -> NodeModel:
        return self._graph.nodes[name][self.MODEL_KEY]

    def __setitem__(self, name: str, model: NodeModel) -> None:
        if name in self._graph.nodes:
            self._graph.nodes[name][self.MODEL_KEY] = model
        else:
            self._graph.add_node(name, model=model)

    @property
    def nodes(self) -> t.Iterable[str]:
        """Returns an iterator over the nodes in the graph."""
        yield from self._graph.nodes

    @property
    def edges(self) -> t.Iterable[t.Tuple[str, str]]:
        """Returns an iterator over the edges in the graph."""
        yield from self._graph.edges

    @property
    def data(self) -> GraphData:
        """Returns the data associated with the graph."""
        return GraphData(
            nodes={n: self[n] for n in self.nodes},
            edges=[(s, t, self.get_spec(s, t)) for s, t in self.edges],
            metadata=self._metadata,
        )

    def add_edge(
        self, source: str, target: str, spec: t.Optional[dataspec.DataSpec] = None
    ) -> None:
        """Add an edge to the graph.

        Args:
            source (str): The source node.
            target (str): The target node.
            spec (t.Optional[DataSpec], optional): The data spec associated with the
                edge. Defaults to None.
        """
        if (source, target) in self._graph.edges:
            self._graph.edges[source, target][self.SPEC_KEY] = spec
        self._graph.add_edge(source, target, spec=spec)

    def get_spec(self, source: str, target: str) -> t.Optional[dataspec.DataSpec]:
        """Get the data spec associated with an edge.

        Args:
            source (str): The source node.
            target (str): The target node.

        Returns:
            t.Optional[DataSpec]: The data spec associated with the edge.
        """
        return self._graph.edges[source, target].get(self.SPEC_KEY, None)

    # TODO: refactor this method
    def collapse(self, depth: int) -> NNGraph:
        """Collapse the graph by grouping nodes at the same level.

        Args:
            depth (int): The depth at which the nodes should be collapsed. If < 0, no
                collapse is performed.

        Returns:
            NNGraph: The collapsed graph.
        """
        if depth < 0:
            return self

        # Create a mapping (path -> node) for each node in the graph
        path_to_node = {}
        for node in self.nodes:
            node_model = self[node]
            path = tuple(node_model.path[:depth])
            path_to_node[path] = node

        # Create a mapping (node -> collapsed node)
        node_to_collapsed: t.Dict[str, t.Tuple[str, NodeModel]] = {}
        collapsed_nodes: t.Dict[str, CollapsedNodeModel] = {}
        for node in self.nodes:
            node_model = self[node]
            if node_model.depth() <= depth:
                node_to_collapsed[node] = node, node_model
            else:
                path = tuple(node_model.path[:depth])
                collapsed_node = path_to_node[path]
                if collapsed_node not in collapsed_nodes:
                    collapsed_nodes[collapsed_node] = CollapsedNodeModel(
                        name=collapsed_node, path=path  # type: ignore
                    )
                collapsed_model = collapsed_nodes[collapsed_node]

                if isinstance(node_model, OpNodeModel):
                    collapsed_model.n_parameters += node_model.n_parameters
                    collapsed_model.perc_parameters += node_model.perc_parameters
                node_to_collapsed[node] = collapsed_node, collapsed_model

        # Create a new graph
        collapsed = self.empty()

        # Populate the new graph
        for source, target in self.edges:
            collapsed_src, model_src = node_to_collapsed[source]
            collapsed_tgt, model_tgt = node_to_collapsed[target]
            collapsed[collapsed_src] = model_src
            collapsed[collapsed_tgt] = model_tgt

            collapsed_spec = None
            if (collapsed_src, collapsed_tgt) in self.edges:
                collapsed_spec = self.get_spec(collapsed_src, collapsed_tgt)
            elif collapsed_tgt != collapsed_src:
                # Find all the edges that start from the collapsed source
                collapsed_src_edges = [
                    (src, tgt)
                    for src, tgt in self.edges
                    if src == collapsed_src and tgt != collapsed_src
                ]

                # Gather specs from all the edges that start from the collapsed source
                collapsed_src_specs = [
                    self.get_spec(src, tgt) for src, tgt in collapsed_src_edges
                ]

                # Find the first spec that is not None
                collapsed_spec = next(
                    (spec for spec in collapsed_src_specs if spec is not None), None
                )

            # Add the edge only if it is not a self-loop
            if collapsed_src != collapsed_tgt:
                collapsed.add_edge(collapsed_src, collapsed_tgt, spec=collapsed_spec)

        return collapsed
