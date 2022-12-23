from abc import ABC, abstractmethod

from nnviz import entities as ent


class GraphDrawer(ABC):
    """An abstract class for drawing a neural network graph. All nnviz graph drawers
    should inherit from this class.
    """

    @abstractmethod
    def draw(self, nngraph: ent.NNGraph) -> None:
        """Draw a neural network graph. This method returns nothing, to avoid making decisions
        about where to save/display/return the graph. The implementation of this interface
        can add custom logic to save/display/return the graph separately.

        Args:
            nngraph (NNGraph): The neural network graph to draw.
        """
        pass
