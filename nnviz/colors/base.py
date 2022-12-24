from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod

color_t = t.Tuple[int, int, int]


class RGBColor:
    """A class that represents a color in RGB format."""

    @classmethod
    def from_hex(cls, hex_color: str) -> RGBColor:
        """Creates a new RGBColor from a hex color in the format #RRGGBB.

        Args:
            hex_color (str): The hex color to convert.

        Returns:
            RGBColor: The RGBColor representation of the hex color.
        """
        r, g, b = tuple(int(hex_color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
        return cls(r, g, b)

    def __init__(self, r: int, g: int, b: int) -> None:
        """Creates a new RGBColor.

        Args:
            r (int): The red component of the color.
            g (int): The green component of the color.
            b (int): The blue component of the color.
        """
        self._color = (r, g, b)

    def is_bright(self) -> bool:
        """Returns whether the color is bright or not.

        Returns:
            bool: True if the color is bright, False otherwise.
        """
        r, g, b = self._color
        return (r * 0.299 + g * 0.587 + b * 0.114) > 186

    def to_hex(self) -> str:
        """Converts the color to a hex color in the format #RRGGBB.

        Returns:
            str: The hex color.
        """
        return "#%02x%02x%02x" % self._color


class ColorPicker(ABC):
    """An abstract class that represents a color picker."""

    @abstractmethod
    def pick(self, *args: t.Hashable) -> RGBColor:
        """Pick a color based on the given arguments.

        Returns:
            RGBColor: The color picked.
        """
        pass
