import typing as t
from pathlib import Path

import typer

app = typer.Typer(name="nnviz")

model_help = "The model to visualize."
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
    import sys
    import subprocess

    import torch.nn as nn
    import torchvision.models as tv_models

    from nnviz import drawing, inspection

    if output_path is None:
        output_path = Path(f"{model}.pdf")

    # TODO: this is a hack to get the model name from the string.
    # This should be done in a better way, maybe using a model registry or something.
    try:
        nn_model: nn.Module = getattr(tv_models, model, None)()  # type: ignore
    except Exception as e:
        raise typer.BadParameter(f"Could not load model {model}") from e

    # TODO: The inspector and drawer should be configurable, not hardcoded
    inspector = inspection.TorchFxInspector()
    drawer = drawing.GraphvizDrawer(output_path)  # What if path is None?

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
