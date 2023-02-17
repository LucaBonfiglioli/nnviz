import importlib
import re
import sys
import typing as t
from pathlib import Path

import torch
import torch.nn as nn
import torchvision

from nnviz.inspection.base import NNInspector
from nnviz.inspection.torchfx import TorchFxInspector


def load_from_torchvision(model: str) -> nn.Module:
    return getattr(torchvision.models, model, None)()  # type: ignore


def load_from_file_import(src: str, symbol_name: str) -> t.Any:
    """Import a symbol from a python file. The file can be everywhere in the file system,
    it will be added to the python path.

    Args:
        src (str): The path to the python file.
        symbol_name (str): The name of the symbol to import.

    Returns:
        t.Any: The imported symbol.
    """
    path = Path(src)
    sys.path.append(str(path.parent))
    module = importlib.import_module(path.stem)
    return getattr(module, symbol_name)


def load_from_module_import(src: str, symbol_name: str) -> t.Any:
    """Import a symbol from a python module. The module must be importable from the
    current python path. The module can be a package.

    Args:
        src (str): The name of the module to import.
        symbol_name (str): The name of the symbol to import.

    Returns:
        t.Any: The imported symbol.
    """
    module = importlib.import_module(src)
    return getattr(module, symbol_name)


def load_from_dynamic_import(model: str) -> t.Any:
    """Import a symbol from a python file or module. The file or module can be anywhere
    in the file system, it will be added to the python path.

    Args:
        model (str): The path to the python file or module, followed by a colon and the

    Returns:
        t.Any: The imported symbol.
    """
    path, _, symbol_name = model.rpartition(":")
    if path.endswith(".py"):
        return load_from_file_import(path, symbol_name)
    else:
        return load_from_module_import(path, symbol_name)


def load_from_string(model: str) -> nn.Module:
    """Load a model from a string. The string can be either the name of a model in

    Args:
        model (str): The name of the model to load.

    Raises:
        ValueError: If the model could not be loaded.

    Returns:
        nn.Module: The loaded model.
    """
    if ":" not in model:
        smb = load_from_torchvision(model)
    else:
        smb = load_from_dynamic_import(model)

    if isinstance(smb, nn.Module):
        return smb

    if callable(smb):
        return smb()

    raise ValueError(f"Could not load model {model}")


DEFAULT_KEY = "x"
"""The default key to use when the input is a single tensor."""


def parse_input_str(in_str: t.Optional[str]) -> t.Optional[t.Dict]:
    """Parse an input string to a dictionary of tensors.

    Args:
        in_str (t.Optional[str]): The input string, can be:

            - None -> None
            - default -> float32 BCHW tensor of shape (1, 3, 224, 224) (commonly used)
            - image<side> (e.g. image224, image256, ...) -> float32 BCHH tensor
            - image<height>x<width> (e.g. image224x224, image256x512, ...) -> float32 BCHW tensor
            - tensor<s0>x<s1>x<s2>x... (e.g. tensor1x3x224x224, tensor1x3x256x512, ...) ->
            float32 generic tensors
            - <key1>:<value1>;<key2>:<value2>;... (e.g. x:tensor1x3x224x224;y:tensor1x3x256x512,
            ...) -> dictionary of tensors

    Returns:
        t.Optional[t.Dict]: A dictionary of tensors to use as input for the model.
    """
    # If the input is None, exit early returning None
    if in_str is None:
        return None

    def _r(istr: str) -> t.Any:
        # Presets syntax
        stx = {
            r"image(\d+)x(\d+)": lambda h, w: torch.rand(1, 3, int(h), int(w)),
            r"image(\d+)": lambda size: torch.rand(1, 3, int(size), int(size)),
            r"tensor(\d+(?:x\d+)*)": lambda shape: torch.rand(
                *[int(s) for s in shape.split("x")]
            ),
            r"((?:\w+):(?:.+);?)+": lambda etr: {
                k: _r(v) for k, v in re.findall(r"(\w+):([^;]+)", etr)
            },
            r"default": lambda: _r("image224"),
        }

        # if in_str matches one of the presets, return the result of the preset passing the
        # variable part of the string to the preset function
        for k, v in stx.items():
            match = re.match(k, istr)
            if match:
                return v(*match.groups())

        # If all else fails -> evil eval
        return eval(istr)

    # Parse the input string
    result = _r(in_str)

    # If the result is not a dict, wrap it in a dict with key "x"
    if not isinstance(result, t.Dict):
        result = {"x": result}

    return result
