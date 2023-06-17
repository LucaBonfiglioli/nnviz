import typing as t
from pathlib import Path

import pydantic
import pygraphviz as pgv

import nnviz.drawing as drawing
from nnviz import colors, dataspec
from nnviz import entities as ent


class HTMLSpecVisitor(dataspec.DataSpecVisitor):
    """Visitor for the `DataSpec` class that produces an HTML table nesting."""

    def __init__(self) -> None:
        """Constructor. Accepts no arguments. and initializes the visitor."""
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


class GraphvizDrawerStyle(pydantic.BaseModel):
    """A style for the GraphvizDrawer."""

    fontname: str = "Arial"
    """The font to use for the graph. Default is "Arial"."""
    default_node_color: str = "gray"
    """The default color for nodes (in case the colorizer fails to return a color). 
    Default is "gray".
    """
    default_edge_color: str = "black"
    """The default color for edges. Default is "black"."""
    node_style: str = "rounded,filled"
    """The style for nodes. See graphviz docs for details. 
    Default is "rounded,filled".
    """
    node_margin: str = "0.2,0.1"
    """The horizontal and vertical margin for nodes. See graphviz docs for details.
    Default is "0.2,0.1"."""
    edge_thickness: str = "2.0"
    """The thickness of edges. Default is "2.0"."""
    graph_title_font_size: int = 48
    """The font size for the graph title. Default is 48."""
    node_title_font_size: int = 24
    """The font size for the node title. Default is 24."""
    cluster_title_font_size: int = 18
    """The font size for the cluster title. Default is 18."""
    show_title: bool = True
    """Whether to show the graph title. Default is True."""
    show_specs: bool = True
    """Whether to show the specs as a label for each edge. Default is True."""
    show_node_name: bool = True
    """Whether to show the node name (just below the title). Default is True."""
    show_node_params: bool = True
    """Whether to show the count of parameters for each node. Default is True."""
    show_node_arguments: bool = True
    """Whether to show the arguments for each node. Default is True."""
    show_node_source: bool = True
    """Whether to show the source of each node. Default is True."""
    show_clusters: bool = True
    """Whether to show the clusters as gray subgraphs. Default is True."""

    def default_graph_params(self) -> t.Dict[str, str]:
        """Returns the default graph parameters."""
        return {
            "fontname": self.fontname,
            "labelloc": "t",
        }

    def default_node_params(self) -> t.Dict[str, str]:
        """Returns the default node parameters."""
        return {
            "fontname": self.fontname,
            "shape": "box",
            "style": self.node_style,
            "margin": self.node_margin,
            "color": self.default_node_color,
            "fillcolor": self.default_node_color,
        }

    def default_edge_params(self) -> t.Dict[str, str]:
        """Returns the default edge parameters."""
        return {
            "fontname": self.fontname,
            "color": self.default_edge_color,
            "penwidth": self.edge_thickness,
        }

    def default_subgraph_params(self) -> t.Dict[str, str]:
        """Returns the default subgraph parameters."""
        return {"fontname": self.fontname, "style": self.node_style}


class GraphvizDrawer(drawing.GraphDrawer):
    """A graph drawer that uses Graphviz to draw a neural network graph."""

    def __init__(
        self,
        path: t.Union[str, Path],
        color_picker: t.Optional[colors.ColorPicker] = None,
        style: t.Optional[GraphvizDrawerStyle] = None,
    ) -> None:
        """Constructor.

        Args:
            path (t.Union[str, Path]): The path to save the graph to. Can be
             a .png, .pdf, .svg file.
            color_picker (t.Optional[colors.ColorPicker], optional): The color picker to
             use for this drawer. If left to none, a color picker will be chosen
             automatically. Defaults to None.
            style (t.Optional[GraphvizDrawerStyle], optional): The style object to use
            see `GraphvizDrawerStyle` for details. If left to none, a default style
            will be chosen automatically. Defaults to None.
        """
        self._path = Path(path)
        self._color_picker = color_picker or colors.BubbleColorPicker()
        self._style = style or GraphvizDrawerStyle()
        self._ignore_prefixes = ["torch.nn."]

    def _text_color(self, color: colors.RGBColor) -> str:
        not_filled = "filled" not in self._style.node_style
        return "black" if color.brightness > 128 or not_filled else "white"

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
        color = rgb.hex
        font_c = self._text_color(rgb)

        title_fsize = self._style.node_title_font_size
        lines = [f'<B><FONT POINT-SIZE="{title_fsize}">{node.op}</FONT></B>']

        if self._style.show_node_name:
            lines.append(f"<I>{node.name}</I>")

        if self._style.show_node_params:
            lines.extend(self._format_params(node))

        if len(node.const_args) > 0 and self._style.show_node_arguments:
            lines.append(
                "<B>args</B>: " + ", ".join([f"{arg}" for arg in node.const_args])
            )

        if len(node.const_kwargs) > 0 and self._style.show_node_arguments:
            kwargs_line = "<B>kwargs</B>: " + ", ".join(
                [f"{key}={value}" for key, value in node.const_kwargs.items()]
            )
            lines.append(kwargs_line)

        if self._style.show_node_source:
            lines.append("")
            lines.append(f"<B>src</B>: {node.full_op}")

        label = self._multi_line(*lines)
        return {"label": label, "color": color, "fillcolor": color, "fontcolor": font_c}

    def _input_output_params(
        self, node: ent.NodeModel, title: str
    ) -> t.Dict[str, t.Any]:
        rgb = colors.RGBColor(0, 0, 0)
        color = rgb.hex
        font_c = self._text_color(rgb)

        fsize = self._style.node_title_font_size
        lines = [f'<B><FONT POINT-SIZE="{fsize}">{title}</FONT></B>']
        if self._style.show_node_name:
            lines.append(f"<I>{node.name}</I>")

        label = self._multi_line(*lines)
        return {"label": label, "color": color, "fillcolor": color, "fontcolor": font_c}

    def _input_params(self, node: ent.InputNodeModel) -> t.Dict[str, t.Any]:
        return self._input_output_params(node, "Input")

    def _output_params(self, node: ent.OutputNodeModel) -> t.Dict[str, t.Any]:
        return self._input_output_params(node, "Output")

    def _collapsed_params(self, node: ent.CollapsedNodeModel) -> t.Dict[str, t.Any]:
        rgb = self._color_picker.pick(*node.path[:-1])
        color = rgb.hex
        font_c = self._text_color(rgb)

        joined_path = ".".join(node.path)
        joined_path = joined_path if len(joined_path) > 0 else "root"
        fsize = self._style.node_title_font_size
        lines = [f'<B><FONT POINT-SIZE="{fsize}">{joined_path}</FONT></B>']

        if self._style.show_node_params:
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
        params = type_map[node.node_type](node)
        return {**self._style.default_node_params(), **params}

    def _subgraph_params(self, name: str, depth: int) -> t.Dict[str, t.Any]:
        # Bg color is a function of depth
        gray_level = int(255 * (1 / (depth / 10 + 1.1)))
        bgcolor = colors.RGBColor(gray_level, gray_level, gray_level).hex

        # Label is the name of the subgraph
        body = name.partition("cluster_")[2]
        fsize = self._style.cluster_title_font_size
        label = f'<<B><FONT POINT-SIZE="{fsize}">{body}</FONT></B>>'

        params = {"label": label, "bgcolor": bgcolor, "labeljust": "l"}
        return {**self._style.default_subgraph_params(), **params}

    def _get_tgt_graph_by_path(
        self, parent: pgv.AGraph, path: t.Sequence[str], depth: int = 0
    ) -> pgv.AGraph:
        # If the path is empty, return the parent graph
        if len(path) <= 1 or not self._style.show_clusters:
            return parent

        # If the subgraph does not exist, create it and recurse
        sg_name = "cluster_" + path[0]
        if parent.get_subgraph(sg_name) is None:
            parent.add_subgraph(name=sg_name, **self._subgraph_params(sg_name, depth))
        target = parent.get_subgraph(sg_name)
        assert target is not None
        return self._get_tgt_graph_by_path(target, path[1:], depth=depth + 1)

    def _edge_params(self, spec: t.Optional[dataspec.DataSpec]) -> t.Dict[str, t.Any]:
        # If the spec is None, or if we don't want to show specs, return the defaults
        if spec is None or not self._style.show_specs:
            return self._style.default_edge_params()

        visitor = HTMLSpecVisitor()
        spec.accept(visitor)

        params = {"label": visitor.html}
        return {**self._style.default_edge_params(), **params}

    def _graph_params(self, nngraph: ent.NNGraph) -> t.Dict[str, t.Any]:
        # If the title is disabled, return the default params only
        if not self._style.show_title:
            return self._style.default_graph_params()

        lines = []
        if nngraph.metadata.title:
            body = nngraph.metadata.title
            fsize = self._style.graph_title_font_size
            lines.append(f'<B><FONT POINT-SIZE="{fsize}">{body}</FONT></B>')

        if nngraph.metadata.source:
            body = f"<B>Source: </B> {nngraph.metadata.source}"
            if nngraph.metadata.source_version:
                body += f" v{nngraph.metadata.source_version}"
            lines.append(body)

        lines.append(f"<B>NNViz </B>v{nngraph.metadata.nnviz_version}")
        lines.append(" ")

        params = {"label": self._multi_line(*lines)}
        return {**self._style.default_graph_params(), **params}

    def _convert(self, nngraph: ent.NNGraph) -> pgv.AGraph:
        # Initialize a pygraphviz graph
        graph_params = self._graph_params(nngraph)
        pgvgraph = pgv.AGraph(directed=True, strict=False, **graph_params)

        # Populate nodes
        for node in sorted(nngraph.nodes):
            model = nngraph[node]
            target_graph = self._get_tgt_graph_by_path(pgvgraph, model.path)
            target_graph.add_node(node, **self._node_params(model))

        # Populate edges
        for source, target, key in nngraph.edges:
            spec = nngraph.get_spec(source, target, key)
            pgvgraph.add_edge(source, target, **self._edge_params(spec))

        return pgvgraph

    def draw(self, nngraph: ent.NNGraph) -> None:
        # Hardcoded to dot, because the other options suck ass.
        prog = "dot"

        # Convert and draw to file
        converted = self._convert(nngraph)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        converted.draw(self._path, prog=prog)
