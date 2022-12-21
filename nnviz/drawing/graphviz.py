from typing import Any, Dict

import pygraphviz as pgv
from hashlib import sha256
from nnviz.entities import (
    NodeModel,
    OpNodeModel,
    ConstantNodeModel,
    CollapsedNodeModel,
    NNGraph,
)
from nnviz.drawing.base import GraphDrawer
from pathlib import Path


class GraphvizDrawer(GraphDrawer):
    def __init__(self, path: Path) -> None:
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
        self._path = path

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

    def _convert(self, nngraph: NNGraph) -> pgv.AGraph:
        # Initialize a pygraphviz graph
        pgvgraph = pgv.AGraph(directed=True, strict=True)

        # Populate nodes
        for node in nngraph.nodes:
            model = nngraph[node]
            pgvgraph.add_node(node, **self._node_params(model))

        # Populate edges
        for source, target in nngraph.edges:
            pgvgraph.add_edge(source, target, **self._default_edge_params)

        return pgvgraph

    def draw(self, nngraph: NNGraph) -> None:
        self._convert(nngraph).draw(self._path, prog="dot")
