# ⭕ Intro

Along with the CLI utility, NNViz can also be used as a **Python library**. This is useful if you want to integrate NNViz into your own Python code, or if you want to use more **advanced features** that are not available in the CLI.

In this section we will cover the **basic concepts** of the NNViz library, which is still rather small and very simple. The things you can do with NNViz are currently limited to:

- **Inspecting** a `nn.Module` and obtain a `NNGraph` object.
- **Manipulating** the `NNGraph` object.
- **Visualizing** the `NNGraph` object to a file.

We will cover each of these steps in the following pages.

## Minimal Example

The following **code snippet** shows the minimal amount of code needed to visualize a `nn.Module`. You can run the code as-is and it will generate a `my_model.pdf` file in the current directory.

```python
import torchvision

from nnviz import drawing, inspection

# User code, for the sake of this example we'll use a torchvision model
my_model = torchvision.models.resnet18()

# Create an inspector: an object that will inspect the model and create a graph
inspector = inspection.TorchFxInspector()

# Inspect the model and draw the graph
graph = inspector.inspect(my_model)

# Create a drawer: an object that will draw the graph to a file
drawer = drawing.GraphvizDrawer("my_model.pdf")

# Draw the graph
drawer.draw(graph)
```