import pytest

from nnviz import colors


class TestRGBColor:
    @pytest.mark.parametrize(
        ["hex", "brightness"],
        [
            ("#000000", 0.0),
            ("#fa21b4", 114.64),
            ("#ffffff", 255.0),
            ("#0000ff", 29.07),
            ("#00ff00", 149.685),
            ("#ff0000", 76.245),
            ("#ffff00", 225.93),
            ("#ddffff", 244.834),
        ],
    )
    def test_rgb_color(self, hex: str, brightness: float):
        color = colors.RGBColor.from_hex(hex)
        assert abs(color.brightness - brightness) < 1e-2
        assert color.hex == hex
