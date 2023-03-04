import typing as t
from pathlib import Path

import pygraphviz as pgv

import nnviz.drawing as drawing
from nnviz import colors, dataspec
from nnviz import entities as ent


class HTMLSpecVisitor(dataspec.DataSpecVisitor):
    """Visitor for the `DataSpec` class that produces an HTML table nesting."""

    def __init__(self) -> None:
        super().__init__()
        self._code = ""
        self._name_stack: t.List[t.List[str]] = []
        self._code_stack: t.List[str] = []

    def _end_visit(self, code: str) -> None:
        if len(self._name_stack) > 0:
            names = self._name_stack[-1]
            name = names.pop(0)  # <--- Pop from the START
            self._code += self._line([name, code])
            if len(names) == 0:
                self._name_stack.pop()
                self._end_composite_visit()
        else:
            self._code += code

    def _table_header(self) -> str:
        return '<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" WIDTH="220%">'

    def _cell(self, text: str) -> str:
        return f"<TD ALIGN='LEFT'>{text}</TD>"

    def _line(self, cells: t.Sequence[str]) -> str:
        return f"<TR>{''.join([self._cell(cell) for cell in cells])}</TR>"

    def _table(self, lines: t.Sequence[t.Sequence[str]]) -> str:
        body = "".join([self._line(line) for line in lines])
        return f"{self._table_header()}{body}</TABLE>"

    def _begin_composite_visit(self, names: t.List[str]) -> None:
        self._name_stack.append(names)
        self._code_stack.append(self._code)
        self._code = self._table_header()

    def _end_composite_visit(self) -> None:
        code = self._code + "</TABLE>"
        self._code = self._code_stack.pop()
        self._end_visit(code)

    def visit_tensor_spec(self, spec: dataspec.TensorSpec) -> None:
        table = self._table(
            [["<B>shape:</B>", str(spec.shape)], ["<B>dtype:</B>", str(spec.dtype)]]
        )
        self._end_visit(table)

    def visit_builtin_spec(self, spec: dataspec.BuiltInSpec) -> None:
        table = self._table([["<B>type:</B>", spec.name]])
        self._end_visit(table)

    def visit_list_spec(self, spec: dataspec.ListSpec) -> None:
        self._begin_composite_visit([str(i) for i in range(len(spec.elements))])

    def visit_map_spec(self, spec: dataspec.MapSpec) -> None:
        self._begin_composite_visit(list(spec.elements.keys()))

    def visit_unknown_spec(self, spec: dataspec.UnknownSpec) -> None:
        self._code += self._table([["Unknown"]])

    @property
    def html(self) -> str:
        """Returns the HTML table."""
        return f"<{self._code}>"


def _human_format(num: t.Union[int, float]) -> str:
    """Returns a human readable format of a number."""
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    if magnitude == 0 and isinstance(num, int):
        return str(num)
    return "%.2f%s" % (num, ["", "K", "M", "G", "T", "P"][magnitude])


class GraphvizDrawer(drawing.GraphDrawer):
    """A graph drawer that uses Graphviz to draw a neural network graph."""

    def __init__(self, path: Path) -> None:
        # TODO: Remove this hard-code rodeo
        self._default_graph_params = {
            "fontname": "Arial",
            "bgcolor": "transparent",
            "labelloc": "t",
        }
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
        self._default_subgraph_params = {
            "fontname": "Arial",
            "fontsize": "12",
            "style": "rounded,filled",
        }
        self._graph_title_size = 48
        self._node_title_size = 24
        self._cluster_title_size = 18
        self._path = path
        # self._color_picker = colors.HashColorPicker()
        self._color_picker = colors.BubbleColorPicker()

        self._ignore_prefixes = [
            "torch.nn.",
        ]

    def _text_color(self, color: colors.RGBColor) -> str:
        return "black" if color.is_bright() else "white"

    def _multi_line(self, *lines: str) -> str:
        multi_line = "<BR/>".join(lines)
        return f"<{multi_line}>"

    def _pick_color_for_op_node(self, node: ent.OpNodeModel) -> colors.RGBColor:
        filtered_op = node.full_op
        # Check if there is a prefix to ignore
        for prefix in self._ignore_prefixes:
            if filtered_op.startswith(prefix):
                filtered_op = filtered_op[len(prefix) :]
                break
        pick_args = filtered_op.split(".") if filtered_op else []

        return self._color_picker.pick(*pick_args)

    def _format_params(
        self, node: t.Union[ent.OpNodeModel, ent.CollapsedNodeModel]
    ) -> t.List[str]:
        lines = []
        if node.n_parameters > 0:
            lines.append("")
            n_params_fmt = _human_format(node.n_parameters)
            perc_params_fmt = _human_format(node.perc_parameters * 100) + "%"
            lines.append(f"<B>params</B>: {n_params_fmt} ({perc_params_fmt})")
        return lines

    def _op_params(self, node: ent.OpNodeModel) -> t.Dict[str, t.Any]:
        rgb = self._pick_color_for_op_node(node)
        color = rgb.to_hex()
        font_c = self._text_color(rgb)

        lines = [
            f'<B><FONT POINT-SIZE="{self._node_title_size}">{node.op}</FONT></B>',
            f"<I>{node.name}</I>",
        ]

        lines.extend(self._format_params(node))

        if len(node.const_args) > 0:
            lines.append(
                "<B>args</B>: " + ", ".join([f"{arg}" for arg in node.const_args])
            )

        if len(node.const_kwargs) > 0:
            kwargs_line = "<B>kwargs</B>: " + ", ".join(
                [f"{key}={value}" for key, value in node.const_kwargs.items()]
            )
            lines.append(kwargs_line)

        lines.append("")
        lines.append(f"<B>src</B>: {node.full_op}")

        label = self._multi_line(*lines)
        return {"label": label, "color": color, "fillcolor": color, "fontcolor": font_c}

    def _input_output_params(
        self, node: ent.NodeModel, title: str
    ) -> t.Dict[str, t.Any]:
        rgb = colors.RGBColor(0, 0, 0)
        color = rgb.to_hex()
        font_c = self._text_color(rgb)

        label = self._multi_line(
            f'<B><FONT POINT-SIZE="{self._node_title_size}">{title}</FONT></B>',
            f"<I>{node.name}</I>",
        )
        return {"label": label, "color": color, "fillcolor": color, "fontcolor": font_c}

    def _input_params(self, node: ent.InputNodeModel) -> t.Dict[str, t.Any]:
        return self._input_output_params(node, "Input")

    def _output_params(self, node: ent.OutputNodeModel) -> t.Dict[str, t.Any]:
        return self._input_output_params(node, "Output")

    def _collapsed_params(self, node: ent.CollapsedNodeModel) -> t.Dict[str, t.Any]:
        rgb = self._color_picker.pick(*node.path[:-1])
        color = rgb.to_hex()
        font_c = self._text_color(rgb)

        joined_path = ".".join(node.path)
        joined_path = joined_path if len(joined_path) > 0 else "root"
        lines = [
            f'<B><FONT POINT-SIZE="{self._node_title_size}">{joined_path}</FONT></B>'
        ]

        lines.extend(self._format_params(node))

        label = self._multi_line(*lines)

        return {"label": label, "color": color, "fillcolor": color, "fontcolor": font_c}

    def _node_params(self, node: ent.NodeModel) -> t.Dict[str, t.Any]:
        type_map = {
            "op": self._op_params,
            "input": self._input_params,
            "output": self._output_params,
            "collapsed": self._collapsed_params,
        }
        params = type_map[node.type_](node)
        return {**self._default_node_params, **params}

    def _subgraph_params(self, name: str, depth: int) -> t.Dict[str, t.Any]:
        # Bg color is a function of depth
        gray_level = int(255 * (1 / (depth / 10 + 1.1)))
        bgcolor = colors.RGBColor(gray_level, gray_level, gray_level).to_hex()

        # Label is the name of the subgraph
        body = name.partition("cluster_")[2]
        label = f'<<B><FONT POINT-SIZE="{self._cluster_title_size}">{body}</FONT></B>>'

        params = {"label": label, "bgcolor": bgcolor, "labeljust": "l"}
        return {**self._default_subgraph_params, **params}

    def _get_tgt_graph_by_path(
        self, parent: pgv.AGraph, path: t.Sequence[str], depth: int = 0
    ) -> pgv.AGraph:
        # If the path is empty, return the parent graph
        if len(path) <= 1:
            return parent

        # If the subgraph does not exist, create it and recurse
        sg_name = "cluster_" + path[0]
        if parent.get_subgraph(sg_name) is None:
            parent.add_subgraph(name=sg_name, **self._subgraph_params(sg_name, depth))
        target = parent.get_subgraph(sg_name)
        assert target is not None
        return self._get_tgt_graph_by_path(target, path[1:], depth=depth + 1)

    def _edge_params(self, spec: t.Optional[dataspec.DataSpec]) -> t.Dict[str, t.Any]:
        if spec is None:
            return self._default_edge_params

        visitor = HTMLSpecVisitor()
        spec.accept(visitor)

        params = {"label": visitor.html}
        return {**self._default_edge_params, **params}

    def _graph_params(self, nngraph: ent.NNGraph) -> t.Dict[str, t.Any]:
        lines = []
        if nngraph.metadata.title:
            body = nngraph.metadata.title
            lines.append(
                f'<B><FONT POINT-SIZE="{self._graph_title_size}">{body}</FONT></B>'
            )

        if nngraph.metadata.source:
            body = f"<B>Source:</B> {nngraph.metadata.source}"
            if nngraph.metadata.source_version:
                body += f" v{nngraph.metadata.source_version}"
            lines.append(body)

        lines.append(f"<B>NNViz </B>v{nngraph.metadata.nnviz_version}")
        lines.append(" ")

        params = {"label": self._multi_line(*lines)}
        return {**self._default_graph_params, **params}

    def _convert(self, nngraph: ent.NNGraph) -> pgv.AGraph:
        # Initialize a pygraphviz graph
        pgvgraph = pgv.AGraph(directed=True, strict=True, **self._graph_params(nngraph))

        # Populate nodes
        for node in nngraph.nodes:
            model = nngraph[node]
            target_graph = self._get_tgt_graph_by_path(pgvgraph, model.path)
            target_graph.add_node(node, **self._node_params(model))

        # Populate edges
        for source, target in nngraph.edges:
            spec = nngraph.get_spec(source, target)
            pgvgraph.add_edge(source, target, **self._edge_params(spec))

        return pgvgraph

    def draw(self, nngraph: ent.NNGraph) -> None:
        converted = self._convert(nngraph)
        prog = "dot"
        converted.draw(self._path, prog=prog)
