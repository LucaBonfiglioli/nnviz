import pytest
import torch.fx as fx
import torchvision


@pytest.fixture(
    params=[
        "resnet18",
        "efficientnet_b0",
        "mobilenet_v2",
        "convnext_tiny",
        "swin_v2_t",
    ]
)
def test_model_name(request) -> str:
    return request.param


@pytest.fixture()
def test_model(test_model_name) -> str:
    return getattr(torchvision.models, test_model_name)(weights="DEFAULT")


@pytest.fixture()
def graph_module(test_model) -> fx.graph_module.GraphModule:
    return fx.symbolic_trace(test_model)  # type: ignore
