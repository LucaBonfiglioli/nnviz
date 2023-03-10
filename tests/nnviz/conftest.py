import pytest
import torch
import torch.fx as fx
import torch.nn as nn
import torchvision

from nnviz import entities as ent
from nnviz import inspection as insp


@pytest.fixture(
    params=["resnet18", "efficientnet_b0", "mobilenet_v2", "convnext_tiny"],
)
def torchvision_model_name(request) -> str:
    return request.param


@pytest.fixture()
def torchvision_model(torchvision_model_name: str) -> str:
    return getattr(torchvision.models, torchvision_model_name)(weights="DEFAULT")


@pytest.fixture()
def graph_module(torchvision_model: nn.Module) -> fx.graph_module.GraphModule:
    return fx.symbolic_trace(torchvision_model)  # type: ignore


@pytest.fixture(params=[None, {"x": torch.randn(1, 3, 224, 224)}])
def nngraph(request, torchvision_model: nn.Module) -> ent.NNGraph:
    return insp.TorchFxInspector().inspect(torchvision_model, inputs=request.param)


@pytest.fixture(params=[-1, 0, 1, 2, 3, 4])
def collapsed_nngraph(nngraph: ent.NNGraph, request) -> ent.NNGraph:
    nngraph.collapse_by_depth(request.param)
    return nngraph
