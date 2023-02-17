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
`torch.nn.Module` with no arguments. Class constructors are considered like functions.
"""
out_help = "The output file path. If not provided, it will save a pdf file named after the model in the current directory."
depth_help = "The maximum depth of the graph. No limit if < 0."
show_help = "Also show the graph after drawing using the default pdf application."
input_help = """The input to feed to the model. If specified, nnviz will also add synthetic
representation of the data passing through the model. Can either be: \n
- "default" -> float32 BCHW tensor of shape (1, 3, 224, 224) (commonly used) \n
- "image<side>" (e.g. image224, image256, ...) -> float32 BCHH tensor \n
- "image<height>x<width>" (e.g. image224x224, image256x512, ...) -> float32 BCHW tensor \n
- "tensor<s0>x<s1>x<s2>x..." (e.g. tensor1x3x224x224, tensor1x3x256x512, ...) ->
float32 generic tensors \n
- "<key1>:<value1>;<key2>:<value2>;... (e.g. x:tensor1x3x224x224;y:tensor1x3x256x512,
...) -> dictionary of tensors \n
- A plain python string that evaluates to a dictionary of tensors (e.g. "{'x': torch.rand(1, 3, 224, 224)}")
"""


@app.command(name="quick")
def quick(
    model: str = typer.Argument(..., help=model_help),
    output_path: t.Optional[Path] = typer.Option(None, "-o", "--out", help=out_help),
    depth: int = typer.Option(1, "-d", "--depth", help=depth_help),
    show: bool = typer.Option(False, "-s", "--show", help=show_help),
    input: str = typer.Option(None, "-i", "--input", help=input_help),
) -> None:
    """Quickly visualize a model."""
    from nnviz import drawing, inspection

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
            typer.echo(f"Could not open {output_path} automatically. I'm sorry.")

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

    # Parse input
    parsed_input = inspection.parse_input_str(input)

    # Inspect
    graph = inspector.inspect(nn_model, inputs=parsed_input)

    # Collapse by depth
    graph = graph.collapse(depth)

    # Draw
    drawer.draw(graph)

    # Show
    if show:
        _show(output_path)


if __name__ == "__main__":
    app()
