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
