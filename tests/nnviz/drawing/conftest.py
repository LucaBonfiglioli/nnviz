from pathlib import Path

import pytest

from nnviz import drawing


@pytest.fixture(params=[True, False])
def style(request):
    return drawing.GraphvizDrawerStyle(
        show_title=request.param,
        show_specs=request.param,
        show_node_name=request.param,
        show_node_params=request.param,
        show_node_arguments=request.param,
        show_node_source=request.param,
        show_clusters=request.param,
    )


@pytest.fixture
def graphviz_drawer(style: drawing.GraphvizDrawerStyle, tmp_path: Path):
    return drawing.GraphvizDrawer(path=tmp_path / "test.pdf", style=style)
