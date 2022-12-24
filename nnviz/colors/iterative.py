import typing as t

from nnviz import colors


class IterativeColorPicker(colors.ColorPicker):
    def __init__(self) -> None:
        super().__init__()

        self._color_tree = {}

    def pick(self, *args: t.Hashable) -> colors.RGBColor:
        raise NotImplementedError("IterativeColorPicker is not implemented yet.")
