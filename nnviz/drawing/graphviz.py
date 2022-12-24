import hashlib
import typing as t
from pathlib import Path

import pygraphviz as pgv

import nnviz.drawing as drawing
from nnviz import entities as ent


class GraphvizDrawer(drawing.GraphDrawer):
    """A graph drawer that uses Graphviz to draw a neural network graph."""

    def __init__(self, path: Path) -> None:
        # TODO: Remove this hard-code rodeo
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
        self._title_size = 24
        self._path = path

    # TODO: this is cursed. Refactor this.
    def _get_color(self, value) -> str:
        hash_ = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
        index = int(hash_, 16) % (2**24)
        r = (index >> 16) & 0xFF
        g = (index >> 8) & 0xFF
        b = index & 0xFF
        return f"#{r:02x}{g:02x}{b:02x}"

    # TODO: this is cursed. Refactor this.
    def _text_color(self, color: str) -> str:
        r, g, b = tuple(int(color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
        return "black" if (r * 0.299 + g * 0.587 + b * 0.114) > 186 else "white"

    def _multi_line(self, *lines: str) -> str:
        multi_line = "<BR/>".join(lines)
        return f"<{multi_line}>"

    def _op_params(self, node: ent.OpNodeModel) -> t.Dict[str, t.Any]:
        color = self._get_color(node.full_op)
        font_c = self._text_color(color)

        lines = [
            f'<B><FONT POINT-SIZE="{self._title_size}">{node.op}</FONT></B>',
            f"<I>{node.name}</I>",
        ]

        if len(node.const_args) > 0:
            lines.append("args: " + ", ".join([f"{arg}" for arg in node.const_args]))

        if len(node.const_kwargs) > 0:
            kwargs_line = "kwargs: " + ", ".join(
                [f"{key}={value}" for key, value in node.const_kwargs.items()]
            )
            lines.append(kwargs_line)

        label = self._multi_line(*lines)
        return {"label": label, "color": color, "fillcolor": color, "fontcolor": font_c}

    def _input_output_params(
        self, node: ent.NodeModel, title: str
    ) -> t.Dict[str, t.Any]:
        color = "#000000"
        font_c = self._text_color(color)

        label = self._multi_line(
            f'<B><FONT POINT-SIZE="{self._title_size}">{title}</FONT></B>',
            f"<I>{node.name}</I>",
        )
        return {"label": label, "color": color, "fillcolor": color, "fontcolor": font_c}

    def _input_params(self, node: ent.InputNodeModel) -> t.Dict[str, t.Any]:
        return self._input_output_params(node, "Input")

    def _output_params(self, node: ent.OutputNodeModel) -> t.Dict[str, t.Any]:
        return self._input_output_params(node, "Output")

    def _collapsed_params(self, node: ent.CollapsedNodeModel) -> t.Dict[str, t.Any]:
        color = self._get_color(node.path[:-1])
        return {
            "label": f'<<B><FONT POINT-SIZE="20">{".".join(node.path)}</FONT></B>>',
            "color": color,
            "fillcolor": color,
            "fontcolor": self._text_color(color),
        }

    def _node_params(self, node: ent.NodeModel) -> t.Dict[str, t.Any]:
        type_map = {
            "op": self._op_params,
            "input": self._input_params,
            "output": self._output_params,
            "collapsed": self._collapsed_params,
        }
        params = type_map[node.type_](node)
        return {**self._default_node_params, **params}

    def _convert(self, nngraph: ent.NNGraph) -> pgv.AGraph:
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

    def draw(self, nngraph: ent.NNGraph) -> None:
        converted = self._convert(nngraph)
        prog = "dot"
        converted.draw(self._path, prog=prog)
