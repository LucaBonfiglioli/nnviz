from pathlib import Path

import pytest

from nnviz import dataspec as ds
from nnviz import entities as ent
from nnviz.drawing import graphviz as nvpgv


class TestGraphvizDrawer:
    def test_draw(self, tmp_path: Path, collapsed_nngraph: ent.NNGraph) -> None:
        drawer = nvpgv.GraphvizDrawer(tmp_path / "test.pdf")
        drawer.draw(collapsed_nngraph)

    @pytest.mark.parametrize("no_title", [True, False])
    @pytest.mark.parametrize("no_source", [True, False])
    @pytest.mark.parametrize("no_source_version", [True, False])
    @pytest.mark.parametrize("torchvision_model_name", ["resnet18"], indirect=True)
    @pytest.mark.parametrize("collapsed_nngraph", [0, 1], indirect=True)
    @pytest.mark.parametrize("nngraph", [None], indirect=True)
    def test_missing_info(
        self,
        tmp_path: Path,
        collapsed_nngraph: ent.NNGraph,
        no_title: bool,
        no_source: bool,
        no_source_version: bool,
    ) -> None:
        if no_title:
            collapsed_nngraph.metadata.title = ""
        if no_source:
            collapsed_nngraph.metadata.source = ""
        if no_source_version:
            collapsed_nngraph.metadata.source_version = ""
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
