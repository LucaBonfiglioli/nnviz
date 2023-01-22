import importlib
import sys
import typing as t
from pathlib import Path

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
