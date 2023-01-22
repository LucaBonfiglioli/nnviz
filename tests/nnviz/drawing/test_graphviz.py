from pathlib import Path

from nnviz import entities as ent
from nnviz.drawing import graphviz


class TestGraphvizDrawer:
    def test_draw(self, tmp_path: Path, collapsed_nngraph: ent.NNGraph) -> None:
        drawer = graphviz.GraphvizDrawer(tmp_path / "test.pdf")
        drawer.draw(collapsed_nngraph)
