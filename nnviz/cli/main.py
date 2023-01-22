import typing as t
from pathlib import Path

import typer

app = typer.Typer(name="nnviz")

model_help = """
The model to visualize. Can either be: \n
- The name of a model in `torchvision.models`. \n
- A string in the form `PATH_TO_FILE:NAME`, where `PATH_TO_FILE` is the path to a python
file containing a model, and `NAME` is the name of the model in that file. Valid names
include objects of type `torch.nn.Module` and functions that return an object of type
`torch.nn.Module` with no arguments. Class constructors are considered like functions.
"""
out_help = "The output file path. If not provided, it will save a pdf file named after the model in the current directory."
depth_help = "The maximum depth of the graph. No limit if < 0."


@app.command(name="quick")
def quick(
    model: str = typer.Argument(..., help=model_help),
    output_path: t.Optional[Path] = typer.Option(None, "-o", "--out", help=out_help),
    depth: int = typer.Option(1, "-d", "--depth", help=depth_help),
    show: bool = typer.Option(
        False, "-s", "--show", help="Show the graph after drawing."
    ),
) -> None:
    """Quickly visualize a model."""

    import os
    import subprocess
    import sys

    from nnviz import drawing, inspection

    if output_path is None:
        _, _, model_name = model.rpartition(":")
        output_path = Path(f"{model_name}.pdf")

    # Load model
    try:
        nn_model = inspection.load_from_string(model)
    except Exception as e:
        raise typer.BadParameter(f"Could not load model {model}") from e

    # TODO: The inspector and drawer should be configurable, not hardcoded
    inspector = inspection.TorchFxInspector()
    drawer = drawing.GraphvizDrawer(output_path)

    # Inspect
    graph = inspector.inspect(nn_model)

    # Collapse by depth
    graph = graph.collapse(depth)

    # Draw
    drawer.draw(graph)

    # Show
    if show:
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
            typer.echo(f"Could not open {output_path} automatically. I'm sorry.")


if __name__ == "__main__":
    app()
