from pathlib import Path
import pytest
from nnviz import inspection as insp
import torch.nn as nn


class StupidModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 1)
        self.fc2 = nn.Linear(1, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


stupid_model = StupidModel()
something_else = 42


@pytest.mark.parametrize(
    "model_str",
    [
        "torchvision.models:resnet18",
        "torchvision.models:efficientnet_b0",
        "torchvision.models:mobilenet_v2",
        "resnet18",
        "efficientnet_b0",
        "mobilenet_v2",
        f"{Path(__file__)}:StupidModel",
        f"{Path(__file__)}:stupid_model",
    ],
)
def test_load_from_string(model_str: str):
    model = insp.load_from_string(model_str)
    assert isinstance(model, nn.Module)


@pytest.mark.parametrize(
    "model_str",
    [
        f"{Path(__file__)}:something_else",
    ],
)
def test_load_from_string_raises(model_str: str):
    with pytest.raises(ValueError):
        insp.load_from_string(model_str)
