from nnviz.entities import NNGraph
from abc import ABC, abstractmethod


class GraphDrawer(ABC):
    @abstractmethod
    def draw(self, nngraph: NNGraph) -> None:
        pass
