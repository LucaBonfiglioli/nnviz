from abc import ABC, abstractmethod
import torch.nn as nn

from nnviz.entities import NNGraph


class NNInspector(ABC):
    @abstractmethod
    def inspect(self, model: nn.Module) -> NNGraph:
        pass
