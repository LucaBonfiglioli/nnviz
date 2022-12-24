import hashlib
import typing as t

from nnviz import colors


class HashColorPicker(colors.ColorPicker):
    """A color picker that uses a hash function to pick a color."""

    def pick(self, *args: t.Hashable) -> colors.RGBColor:
        hash_ = hashlib.sha256(str(args).encode("utf-8")).hexdigest()
        index = int(hash_, 16) % (2**24)
        r = (index >> 16) & 0xFF
        g = (index >> 8) & 0xFF
        b = index & 0xFF
        return colors.RGBColor(r, g, b)
