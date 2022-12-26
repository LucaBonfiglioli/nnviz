import pytest

from nnviz import colors


class TestRGBColor:
    @pytest.mark.parametrize(
        ["hex", "bright"],
        [
            ("#000000", False),
            ("#fa21b4", False),
            ("#ffffff", True),
            ("#0000ff", False),
            ("#00ff00", False),
            ("#ff0000", False),
            ("#ffff00", True),
            ("#ddffff", True),
        ],
    )
    def test_rgb_color(self, hex: str, bright: bool):
        color = colors.RGBColor.from_hex(hex)
        assert color.is_bright() == bright
        assert color.to_hex() == hex
