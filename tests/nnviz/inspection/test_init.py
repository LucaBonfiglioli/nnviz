from pathlib import Path

import pytest
import torch.nn as nn

from nnviz import dataspec as ds
from nnviz import inspection as insp


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
        "resnet18;pretrained=True",
        "resnet18;True",
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


@pytest.mark.parametrize("prefix", ["", "x:", "y:", "abcd:"])
@pytest.mark.parametrize(
    ["in_str", "expected"],
    [
        ["default", ds.TensorSpec(shape=(1, 3, 224, 224), dtype="torch.float32")],
        ["image32", ds.TensorSpec(shape=(1, 3, 32, 32), dtype="torch.float32")],
        ["image32x32", ds.TensorSpec(shape=(1, 3, 32, 32), dtype="torch.float32")],
        ["image32x64", ds.TensorSpec(shape=(1, 3, 32, 64), dtype="torch.float32")],
        ["tensor32", ds.TensorSpec(shape=(32,), dtype="torch.float32")],
        ["tensor32x32", ds.TensorSpec(shape=(32, 32), dtype="torch.float32")],
        ["tensor32x64", ds.TensorSpec(shape=(32, 64), dtype="torch.float32")],
        ["tensor32x64x128", ds.TensorSpec(shape=(32, 64, 128), dtype="torch.float32")],
        [
            "torch.randn(30, 52).int() + 42 # I love Jesus",
            ds.TensorSpec(shape=(30, 52), dtype="torch.int32"),
        ],
    ],
)
def test_parse_input_str_single(in_str: str, prefix: str, expected: ds.DataSpec):
    expected_key = prefix[:-1] if prefix else insp.DEFAULT_KEY
    spec = ds.DataSpec.build(insp.parse_input_str(prefix + in_str))
    assert spec == ds.MapSpec(elements={expected_key: expected})


@pytest.mark.parametrize(
    ["in_str", "expected"],
    [
        [
            "x:default;y:default",
            ds.MapSpec(
                elements={
                    "x": ds.TensorSpec(shape=(1, 3, 224, 224), dtype="torch.float32"),
                    "y": ds.TensorSpec(shape=(1, 3, 224, 224), dtype="torch.float32"),
                }
            ),
        ],
        [
            "x:default;y:image32;abcd:tensor1x3x32x64",
            ds.MapSpec(
                elements={
                    "x": ds.TensorSpec(shape=(1, 3, 224, 224), dtype="torch.float32"),
                    "y": ds.TensorSpec(shape=(1, 3, 32, 32), dtype="torch.float32"),
                    "abcd": ds.TensorSpec(shape=(1, 3, 32, 64), dtype="torch.float32"),
                }
            ),
        ],
    ],
)
def test_parse_input_str_multi(in_str: str, expected: ds.DataSpec):
    assert ds.DataSpec.build(insp.parse_input_str(in_str)) == expected


def test_parse_input_str_none():
    assert insp.parse_input_str(None) is None
