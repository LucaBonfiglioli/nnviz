from abc import ABC, abstractmethod

import torch.nn as nn

from nnviz import entities as ent


class NNInspector(ABC):
    """An abstract class for inspecting a neural network model. All nnviz inspectors
    should inherit from this class.
    """

    @abstractmethod
    def inspect(self, model: nn.Module) -> ent.NNGraph:
        """Inspect a neural network model and return a NNGraph object.

        Args:
            model (nn.Module): The neural network model to inspect.

        Returns:
            NNGraph: The graph representation of the neural network model.
        """
        pass
