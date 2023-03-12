import typing as t
from pathlib import Path

import typer

app = typer.Typer(name="nnviz", pretty_exceptions_enable=False)

model_help = """
The model to visualize. Can either be: \n
- The name of a model in `torchvision.models`. \n
- A string in the form `PATH_TO_FILE:NAME`, where `PATH_TO_FILE` is the path to a python
file containing a model, and `NAME` is the name of the model in that file. Valid names
include objects of type `torch.nn.Module` and functions that return an object of type
`torch.nn.Module` with no arguments. Class constructors are considered like functions.\n
- A PATH to a json file containing a previously serialized graph. In this case NNViz
will simply load the graph (without inspecting anything) and draw it. To create such a
file, use the `-j` or `--json` flag when running this command.\n
"""
out_help = """
The output file path. If not provided, it will save a pdf file named after the model in
the current directory.
"""
depth_help = "The maximum depth of the graph. No limit if < 0. Default is 2."
show_help = "Also show the graph after drawing using the default pdf application."
input_help = """The input to feed to the model. If specified, nnviz will also add
synthetic representation of the data passing through the model. Can either be: \n
- "default" -> float32 BCHW tensor of shape (1, 3, 224, 224) (commonly used) \n
- "image<side>" (e.g. image224, image256, ...) -> float32 BCHH tensor \n
- "image<height>x<width>" (e.g. image224x224, image32x64, ...) -> float32 BCHW tensor\n
- "tensor<s0>x<s1>x<s2>x..." (e.g. tensor1x3x224x224, tensor1x3x256x512, ...) ->
float32 generic tensors \n
- "<key1>:<value1>;<key2>:<value2>;... (e.g. x:tensor1x3x224x224;y:tensor1x3x256x512,
...) -> dictionary of tensors \n
- A plain python string that evaluates to a dictionary of tensors
(e.g. "{'x': torch.rand(1, 3, 224, 224)}")
"""
layer_help = """
The name of the layer to visualize. If not provided, the whole model
will be visualized.
"""
json_help = "Also save the graph as a json file, named just like the output file."
collapse_help = """
Layers that should collapsed, besides the ones that are collapsed by depth.
"""
quiet_help = "Disable all printing besides eventual errors."
style_help = """
List of style options to apply, in the form `key=value`. Don't worry about the type of
the value, it will be automatically inferred by pydantic. The available options are: \n
- fontname: str - The font to use for the graph. Default is "Arial". \n
- default_node_color: str - The default color for nodes (in case the colorizer fails to
return a color). Default is "gray". \n
- default_edge_color: str - The default color for edges. Default is "black". \n
- node_style: str - The style for nodes. See graphviz docs for details. Default is
"rounded,filled". \n
- node_margin: str - The horizontal and vertical margin for nodes. See graphviz docs for
details. Default is "0.2,0.1". \n
- edge_thickness: str - The thickness of edges. Default is "2.0". \n
- graph_title_font_size: int - The font size for the graph title. Default is 48. \n
- node_title_font_size: int - The font size for the node title. Default is 24. \n
- cluster_title_font_size: int - The font size for the cluster title. Default is 18. \n
- show_title: bool - Whether to show the graph title. Default is True. \n
- show_specs: bool - Whether to show the specs as a label for each edge. 
Default is True. \n
- show_node_name: bool - Whether to show the node name (just below the title).  
Default is True. \n
- show_node_params: bool - Whether to show the count of parameters for each node. 
Default is True. \n
- show_node_arguments: bool - Whether to show the arguments for each node. 
Default is True. \n
- show_node_source: bool - Whether to show the source of each node. Default is True. \n
- show_clusters: bool - Whether to show the clusters as gray subgraphs. 
Default is True. \n
"""


@app.command(name="quick")
def quick(
    # Arguments
    model: str = typer.Argument(..., help=model_help),
    # Options
    layer: t.Optional[str] = typer.Option(None, "-l", "--layer", help=layer_help),
    input: str = typer.Option(None, "-i", "--input", help=input_help),
    depth: int = typer.Option(2, "-d", "--depth", help=depth_help),
    collapse: t.List[str] = typer.Option([], "-c", "--collapse", help=collapse_help),
    output: t.Optional[Path] = typer.Option(None, "-o", "--out", help=out_help),
    # Style
    style: t.List[str] = typer.Option([], "-S", "--style", help=style_help),
    # Flags
    show: bool = typer.Option(False, "-s", "--show", help=show_help),
    json: bool = typer.Option(False, "-j", "--json", help=json_help),
    quiet: bool = typer.Option(False, "-q", "--quiet", help=quiet_help),
) -> None:
    """Quickly visualize a model."""
    from traceback import print_exc

    from nnviz import drawing, entities, inspection

    def _show(output_path: Path) -> None:
        import os
        import subprocess
        import sys

        # Windows
        if os.name == "nt":
            os.startfile(output_path, "open")

        # Mac
        elif sys.platform == "darwin":
            subprocess.Popen(["open", output_path])

        # Linux
        elif sys.platform.startswith("linux"):
            subprocess.Popen(["xdg-open", output_path])

        else:
            RuntimeError("Could not automatically open the file.")

    if output is None:
        _, _, model_name = model.rpartition(":")
        if layer is not None:
            model_name += f"_{layer}".replace(".", "-")
        output = Path(model_name).with_suffix(".pdf")

    # If the model is a json file, load it directly without inspecting
    if model.endswith(".json"):
        if not quiet:
            typer.echo(f"Loading graph from json file ({model})...")
        try:
            graph_data = entities.GraphData.parse_file(model)
            graph = entities.NNGraph.from_data(graph_data)
        except Exception as e:
            print_exc()
            raise typer.BadParameter(
                f"Could not load graph from json file {model}"
            ) from e

        # Remove .json from model name (mildly cursed)
        model = model[:-5]

    # Otherwise, inspect the model
    else:
        # Load model
        if not quiet:
            typer.echo(f"Parsing model '{model}'...")
        try:
            nn_model = inspection.load_from_string(model)
        except Exception as e:
            print_exc()
            raise typer.BadParameter(f"Could not load model {model}") from e

        # Get layer if needed
        if layer is not None:
            try:
                nn_model = nn_model.get_submodule(layer)
            except Exception as e:
                raise typer.BadParameter(f"Could not find layer {layer}") from e

        # Create inspector
        inspector = inspection.TorchFxInspector()

        # Parse input
        if not quiet and input is not None:
            typer.echo(f"Parsing input '{input}'...")
        try:
            parsed_input = inspection.parse_input_str(input)
        except Exception as e:
            print_exc()
            raise typer.BadParameter(f"Could not parse input {input}") from e

        # Inspect
        if not quiet:
            typer.echo("Inspecting model...")
        try:
            graph = inspector.inspect(nn_model, inputs=parsed_input)
        except Exception as e:
            print_exc()
            raise typer.BadParameter(f"Could not inspect model {model}") from e

    # Save json if needed
    if json:
        json_path = output.with_suffix(".json")
        if not quiet:
            typer.echo(f"Saving graph as json file ({json_path})...")
        try:
            with open(json_path, "w") as f:
                f.write(graph.data.json())
        except Exception as e:
            print_exc()
            raise typer.BadParameter(
                f"Could not save graph as json file {json_path}"
            ) from e

    if not quiet and (depth > 0 or len(collapse) > 0):
        typer.echo("Collapsing graph...")
    try:
        # Collapse by depth
        graph.collapse_by_depth(depth)

        # Collapse by name
        graph.collapse_multiple(collapse)
    except Exception as e:
        print_exc()
        raise typer.BadParameter("Could not collapse graph") from e

    # Draw
    if not quiet:
        typer.echo(f"Drawing graph to {output}...")
    try:
        parsed_style_dict = {}
        for s in style:
            key, _, value = s.partition("=")
            parsed_style_dict[key] = value
        parsed_style = drawing.GraphvizDrawerStyle.parse_obj(parsed_style_dict)
        drawer = drawing.GraphvizDrawer(output, style=parsed_style)
        drawer.draw(graph)
    except Exception as e:
        print_exc()
        raise typer.BadParameter(f"Could not draw graph to {output}") from e

    # Show
    if show:
        if not quiet:
            typer.echo(f"Opening {output}...")
        try:
            _show(output)
        except Exception as e:
            print_exc()
            raise typer.BadParameter(f"Could not open {output}") from e

    if not quiet:
        typer.echo()
        typer.echo("Done!")


if __name__ == "__main__":
    app()
