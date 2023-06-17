from __future__ import annotations

import typing as t

import networkx as nx
import pydantic as pyd

import nnviz
from nnviz import dataspec


class NodeModel(pyd.BaseModel):
    """Pydantic model for a node in the graph. Contains some information about a layer
    of a neural network."""

    node_type: t.Literal[""] = ""
    """NodeModel type discriminator. Do not set this field manually."""

    name: str = pyd.Field(...)
    """Qualified name of the node."""

    path: t.Sequence[str] = pyd.Field(default_factory=list)
    """Path of the node. E.g. ['features', '0', 'conv1']"""


class OpNodeModel(NodeModel):
    """`NodeModel` specialized for an operation node."""

    node_type: t.Literal["op"] = "op"
    """NodeModel type discriminator. Do not set this field manually."""

    op: str = ""
    """Name of the operation. E.g. 'Conv2d'."""

    full_op: str = ""
    """Full name of the operation. E.g. 'torch.nn.modules.conv.Conv2d'."""

    const_args: t.Sequence[t.Any] = pyd.Field(default_factory=list)
    """Constant positional arguments of the operation."""

    const_kwargs: t.Mapping[str, t.Any] = pyd.Field(default_factory=dict)
    """Constant keyword arguments of the operation."""

    n_parameters: int = 0
    """Number of parameters of the operation."""

    perc_parameters: float = 0.0
    """Percentage of parameters of the operation."""


class CollapsedNodeModel(NodeModel):
    """`NodeModel` specialized for a collapsed node."""

    node_type: t.Literal["collapsed"] = "collapsed"
    """NodeModel type discriminator. Do not set this field manually."""

    n_parameters: int = 0
    """Number of parameters of the collapsed node."""

    perc_parameters: float = 0.0
    """Percentage of parameters of the collapsed node."""


class InputNodeModel(NodeModel):
    """`NodeModel` specialized for an input node."""

    node_type: t.Literal["input"] = "input"
    """NodeModel type discriminator. Do not set this field manually."""


class OutputNodeModel(NodeModel):
    """`NodeModel` specialized for an output node."""

    node_type: t.Literal["output"] = "output"
    """NodeModel type discriminator. Do not set this field manually."""


t_any_node = t.Union[
    NodeModel, OpNodeModel, CollapsedNodeModel, InputNodeModel, OutputNodeModel
]


class GraphMeta(pyd.BaseModel):
    """Pydantic model for the metadata associated with the graph."""

    title: str = ""
    """Title of the graph."""

    source: str = ""
    """Where the graph comes from."""

    source_version: str = ""
    """Graph source library version."""

    nnviz_version: str = pyd.Field(nnviz.__version__)
    """Version of nnviz used to generate the graph."""


class GraphData(pyd.BaseModel):
    """Pydantic model for the data associated with the graph. It consists of an anemic
    representation of the graph, i.e. a mapping of nodes and a list of edges.
    This is done to make it easier to serialize and deserialize the graph, without
    having to deal with the networkx graph representation.
    """

    nodes: t.Mapping[str, t_any_node] = pyd.Field(default_factory=dict)
    """Mapping of nodes names to their models."""

    edges: t.Sequence[
        t.Tuple[str, str, str, t.Optional[dataspec.t_any_spec]]
    ] = pyd.Field(default_factory=list)
    """List of edges. As tuples (src, tgt, key, [spec])."""

    metadata: GraphMeta = GraphMeta()  # type: ignore
    """Metadata associated with the graph."""


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
        for source, target, key, spec in data.edges:
            graph.add_edge(source, target, key, spec=spec)

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
        """Returns the model associated with the node name."""
        return self._graph.nodes[name][self.MODEL_KEY]

    def __setitem__(self, name: str, model: NodeModel) -> None:
        """Sets the model associated with the node name. And, if the node does not
        exist, it is created.
        """
        if name in self._graph.nodes:
            self._graph.nodes[name][self.MODEL_KEY] = model
        else:
            self._graph.add_node(name, model=model)

    @property
    def nodes(self) -> t.Iterable[str]:
        """Returns an iterator over the nodes in the graph."""
        yield from self._graph.nodes

    @property
    def edges(self) -> t.Iterable[t.Tuple[str, str, str]]:
        """Returns an iterator over the edges in the graph."""
        yield from self._graph.edges

    @property
    def data(self) -> GraphData:
        """Returns the data associated with the graph."""
        return GraphData(
            nodes={n: self[n] for n in self.nodes},
            edges=[(s, t, k, self.get_spec(s, t, k)) for s, t, k in self.edges],
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
        self,
        source: str,
        target: str,
        key: str,
        spec: t.Optional[dataspec.DataSpec] = None,
    ) -> None:
        """Add an edge to the graph.

        Args:
            source (str): The source node.
            target (str): The target node.
            key (str): The key of the edge.
            spec (t.Optional[DataSpec], optional): The data spec associated with the
                edge. Defaults to None.
        """
        self._graph.add_edge(source, target, key, spec=spec)

    def get_spec(
        self, source: str, target: str, key: str
    ) -> t.Optional[dataspec.DataSpec]:
        """Get the data spec associated with an edge.

        Args:
            source (str): The source node.
            target (str): The target node.
            key (str): The key of the edge.

        Returns:
            t.Optional[DataSpec]: The data spec associated with the edge.
        """
        return self._graph.edges[source, target, key].get(self.SPEC_KEY, None)

    def collapse(self, path: t.Sequence[str]) -> None:
        """Garther all nodes sharing the same path prefix and collapse them into a
        single collapsed node. This operation is done in place and is not reversible.

        Args:
            path (t.Sequence[str]): The path prefix to collapse.
        """
        # Find all nodes sharing the same path prefix
        to_coll = [n for n in self.nodes if tuple(self[n].path[: len(path)]) == path]

        # Gather their models
        models = [self[n] for n in to_coll]

        # [Get number of parameters for each node]
        op_models = [m for m in models if isinstance(m, OpNodeModel)]
        n_params = sum([n.n_parameters for n in op_models])
        perc_params = sum([n.perc_parameters for n in op_models])

        # Find all incoming edges to these nodes (must come from outside the group)
        _t = t.Set[t.Tuple[str, str, str]]
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
            self.add_edge(source, collapsed, key, spec=spec)

        # Add edges from the collapsed node to the outgoing edges
        for source, target, key in out_edges:
            spec = self.get_spec(source, target, key)
            self.add_edge(collapsed, target, key, spec=spec)

        # Delete all edges in the group
        self._graph.remove_edges_from(self_edges)
        self._graph.remove_edges_from(in_edges)
        self._graph.remove_edges_from(out_edges)

        # Delete all nodes in the group
        self._graph.remove_nodes_from(to_coll)

    def collapse_multiple(
        self, paths: t.Sequence[t.Union[t.Sequence[str], str]]
    ) -> None:
        """Collapse multiple paths in the graph.

        Args:
            paths (t.Sequence[t.Union[Sequence[str], str]]): The paths to collapse.
        """
        # Convert all paths to tuples
        paths = [p.split(".") if isinstance(p, str) else p for p in paths]

        # Remove duplicates
        unique_paths = {tuple(p) for p in paths}

        # If a path is a prefix of another path, keep only the shortest one (i.e. the
        # one with the least number of elements)
        unique_paths = {
            p
            for p in unique_paths
            if not any([p[: len(p2)] == p2 for p2 in unique_paths.difference({p})])
        }

        # Collapse each group using the `collapse` methods
        for path in unique_paths:
            self.collapse(path)

    def collapse_by_lambda(
        self, func: t.Callable[[NodeModel], t.Sequence[str]]
    ) -> None:
        """Collapse the graph by grouping nodes that satisfy a given condition.

        Args:
            func (t.Callable[[NodeModel], t.Sequence[str]]): A function that
                takes a node and returns the path it should be collapsed to. If the
                returned path is a prefix of the original path, the node is grouped with
                the other nodes that have the same prefix.
        """
        # Collapse all nodes that have the same prefix
        reduced = {n: func(self[n]) for n in self.nodes}
        self.collapse_multiple([v for k, v in reduced.items() if self[k].path != v])

    def collapse_by_depth(self, depth: int) -> None:
        """Collapse the graph by grouping nodes at the same level.

        Args:
            depth (int): The depth at which the nodes should be collapsed. If < 1, no
                collapse is performed.
        """
        if depth < 1:
            return

        self.collapse_by_lambda(lambda p: p.path[:depth])
