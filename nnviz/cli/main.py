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
depth_help = "The maximum depth of the graph. No limit if < 0."
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
json_help = "Save the graph as a json file instead of a pdf."
collapse_help = """
Layers that should collapsed, besides the ones that are collapsed by depth.
"""
quiet_help = "Disable all printing besides eventual errors."


@app.command(name="quick")
def quick(
    # Arguments
    model: str = typer.Argument(..., help=model_help),
    # Options
    layer: t.Optional[str] = typer.Option(None, "-l", "--layer", help=layer_help),
    input: str = typer.Option(None, "-i", "--input", help=input_help),
    depth: int = typer.Option(1, "-d", "--depth", help=depth_help),
    collapse: t.List[str] = typer.Option([], "-c", "--collapse", help=collapse_help),
    output: t.Optional[Path] = typer.Option(None, "-o", "--out", help=out_help),
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
        output = Path(f"{model_name}.pdf")

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
        drawer = drawing.GraphvizDrawer(output)
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
