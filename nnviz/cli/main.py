# Mock model for testing
from pathlib import Path
from torchvision.models import efficientnet_b0
from nnviz.inspection.torchfx import TorchFxInspector
from nnviz.drawing.graphviz import GraphvizDrawer

if __name__ == "__main__":
    to_insepct = efficientnet_b0()

    inspector = TorchFxInspector()
    G = inspector.inspect(to_insepct)

    # Collapse the graph
    import sys

    G = G.collapse(int(sys.argv[1]))

    # Draw the graph
    drawer = GraphvizDrawer(Path(f"test{sys.argv[1]}.pdf"))
    drawer.draw(G)
