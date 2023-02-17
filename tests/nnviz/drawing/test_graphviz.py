from pathlib import Path

import pytest

from nnviz import dataspec as ds
from nnviz import entities as ent
from nnviz.drawing import graphviz as nvpgv


class TestGraphvizDrawer:
    def test_draw(self, tmp_path: Path, collapsed_nngraph: ent.NNGraph) -> None:
        drawer = nvpgv.GraphvizDrawer(tmp_path / "test.pdf")
        drawer.draw(collapsed_nngraph)


class TestHTMLSpecVisitor:
    @pytest.mark.parametrize(
        "spec",
        [
            ds.UnknownSpec(),
            ds.BuiltInSpec(name="int"),
            ds.TensorSpec(shape=[1, 2, 3], dtype="float32"),
            ds.ListSpec(elements=[ds.TensorSpec(shape=[1, 2, 3], dtype="float32")]),
            ds.MapSpec(
                elements={
                    "a": ds.TensorSpec(shape=[1, 2, 3], dtype="float32"),
                    "b": ds.TensorSpec(shape=[1, 2, 3], dtype="float32"),
                }
            ),
        ],
    )
    def test_visit(self, spec: ds.DataSpec) -> None:
        visitor = nvpgv.HTMLSpecVisitor()

        spec.accept(visitor)
        assert visitor.html
