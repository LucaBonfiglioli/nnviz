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
    source: str = pyd.Field("", description="Where the graph comes from.")
    source_version: str = pyd.Field("", description="Graph source version.")
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
        return cls(nx.MultiDiGraph(), GraphMeta.parse_obj({}))

    @classmethod
    def from_data(cls, data: GraphData) -> NNGraph:
        """Create a graph from the data associated with it.

        Args:
            data (GraphData): Data to create the graph from.

        Returns:
            NNGraph: The graph with fully populated nodes and edges.
        """
        graph = cls(nx.MultiDiGraph(), data.metadata)
        for name, model in data.nodes.items():
            graph[name] = model
        for source, target, spec in data.edges:
            graph.add_edge(source, target, spec=spec)

        return graph

    def __init__(self, graph: nx.MultiDiGraph, metadata: GraphMeta) -> None:
        """Constructor.

        Args:
            graph (nx.MultiDiGraph): NetworkX graph wrapped by this class.
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
    def edges(self) -> t.Iterable[t.Tuple[str, str, int]]:
        """Returns an iterator over the edges in the graph."""
        yield from self._graph.edges

    @property
    def data(self) -> GraphData:
        """Returns the data associated with the graph."""
        return GraphData(
            nodes={n: self[n] for n in self.nodes},
            edges=[(s, t, self.get_spec(s, t, k)) for s, t, k in self.edges],
            metadata=self._metadata,
        )

    @property
    def metadata(self) -> GraphMeta:
        """Returns the metadata associated with the graph."""
        return self._metadata

    @metadata.setter
    def metadata(self, metadata: GraphMeta) -> None:
        """Sets the metadata associated with the graph."""
        self._metadata = metadata

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
        self._graph.add_edge(source, target, spec=spec)

    def get_spec(
        self, source: str, target: str, key: int
    ) -> t.Optional[dataspec.DataSpec]:
        """Get the data spec associated with an edge.

        Args:
            source (str): The source node.
            target (str): The target node.
            key (int): The key of the edge.

        Returns:
            t.Optional[DataSpec]: The data spec associated with the edge.
        """
        return self._graph.edges[source, target, key].get(self.SPEC_KEY, None)

    def collapse(self, path: t.Sequence[str]) -> None:
        # Find all nodes sharing the same path prefix
        to_coll = [n for n in self.nodes if tuple(self[n].path[: len(path)]) == path]

        # If there are less than 2 nodes, there is nothing to collapse, so return
        if len(to_coll) < 2:
            return

        # Gather their models
        models = [self[n] for n in to_coll]

        # [Get number of parameters for each node]
        op_models = [m for m in models if isinstance(m, OpNodeModel)]
        n_params = sum([n.n_parameters for n in op_models])
        perc_params = sum([n.perc_parameters for n in op_models])

        # Find all incoming edges to these nodes (must come from outside the group)
        _t = t.Set[t.Tuple[str, str, int]]
        in_edges: _t = set(self._graph.in_edges(to_coll, keys=True))  # type: ignore

        # Find all outgoing edges to these nodes (must go outside the group)
        out_edges: _t = set(self._graph.out_edges(to_coll, keys=True))  # type: ignore

        # Compute the intersection of the incoming and outgoing edges and remove them
        # from the list of incoming and outgoing edges
        self_edges = in_edges.intersection(out_edges)
        in_edges -= self_edges
        out_edges -= self_edges

        # Add a single collapsed node with the same path prefix
        collapsed = "_".join(path)
        collapsed_model = CollapsedNodeModel(
            name=collapsed,
            path=path,
            n_parameters=n_params,
            perc_parameters=perc_params,
        )
        self[collapsed] = collapsed_model

        # Add edges from the incoming edges to the collapsed node
        for source, target, key in in_edges:
            spec = self.get_spec(source, target, key)
            self.add_edge(source, collapsed, spec=spec)

        # Add edges from the collapsed node to the outgoing edges
        for source, target, key in out_edges:
            spec = self.get_spec(source, target, key)
            self.add_edge(collapsed, target, spec=spec)

        # Delete all edges in the group
        self._graph.remove_edges_from(self_edges)
        self._graph.remove_edges_from(in_edges)
        self._graph.remove_edges_from(out_edges)

        # Delete all nodes in the group
        self._graph.remove_nodes_from(to_coll)

    def collapse_by_depth(self, depth: int) -> None:
        """Collapse the graph by grouping nodes at the same level.

        Args:
            depth (int): The depth at which the nodes should be collapsed. If < 1, no
                collapse is performed.
        """
        if depth < 1:
            return

        # Get all node paths
        paths = [tuple(self[n].path) for n in self.nodes]

        # Group nodes by path prefix
        paths = {p[:depth] for p in paths}

        # Collapse each group using the `collapse` methods
        for path in paths:
            self.collapse(path)

        # # Create a mapping (path -> node) for each node in the graph
        # path_to_node = {}
        # for node in self.nodes:
        #     node_model = self[node]
        #     path = tuple(node_model.path[:depth])
        #     path_to_node[path] = node

        # # Create a mapping (node -> collapsed node)
        # node_to_collapsed: t.Dict[str, t.Tuple[str, NodeModel]] = {}
        # collapsed_nodes: t.Dict[str, CollapsedNodeModel] = {}
        # for node in self.nodes:
        #     node_model = self[node]
        #     if node_model.depth() <= depth:
        #         node_to_collapsed[node] = node, node_model
        #     else:
        #         path = tuple(node_model.path[:depth])
        #         collapsed_node = path_to_node[path]
        #         if collapsed_node not in collapsed_nodes:
        #             collapsed_nodes[collapsed_node] = CollapsedNodeModel(
        #                 name=collapsed_node, path=path  # type: ignore
        #             )
        #         collapsed_model = collapsed_nodes[collapsed_node]

        #         if isinstance(node_model, OpNodeModel):
        #             collapsed_model.n_parameters += node_model.n_parameters
        #             collapsed_model.perc_parameters += node_model.perc_parameters
        #         node_to_collapsed[node] = collapsed_node, collapsed_model

        # # Create a new graph
        # collapsed = self.empty()
        # collapsed.metadata = self.metadata

        # # Populate the new graph
        # for source, target in self.edges:
        #     collapsed_src, model_src = node_to_collapsed[source]
        #     collapsed_tgt, model_tgt = node_to_collapsed[target]
        #     collapsed[collapsed_src] = model_src
        #     collapsed[collapsed_tgt] = model_tgt

        #     collapsed_spec = None
        #     if (collapsed_src, collapsed_tgt) in self.edges:
        #         collapsed_spec = self.get_spec(collapsed_src, collapsed_tgt)
        #     elif collapsed_tgt != collapsed_src:
        #         # Find all the edges that start from the collapsed source
        #         collapsed_src_edges = [
        #             (src, tgt)
        #             for src, tgt in self.edges
        #             if src == collapsed_src and tgt != collapsed_src
        #         ]

        #         # Gather specs from all the edges that start from the collapsed source
        #         collapsed_src_specs = [
        #             self.get_spec(src, tgt) for src, tgt in collapsed_src_edges
        #         ]

        #         # Find the first spec that is not None
        #         collapsed_spec = next(
        #             (spec for spec in collapsed_src_specs if spec is not None), None
        #         )

        #     # Add the edge only if it is not a self-loop
        #     if collapsed_src != collapsed_tgt:
        #         collapsed.add_edge(collapsed_src, collapsed_tgt, spec=collapsed_spec)

        # return collapsed
