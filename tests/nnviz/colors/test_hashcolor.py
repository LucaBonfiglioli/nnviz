import pytest

from nnviz import colors


class TestHashColorPicker:
    @pytest.mark.parametrize(
        "args",
        [
            (1, 2, 3),
            ("a", "b", "c"),
            (1, "a", 2, "b"),
            (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
            tuple(),
        ],
    )
    def test_pick(self, args):
        picker = colors.HashColorPicker()
        color = picker.pick(*args)
        assert isinstance(color, colors.RGBColor)

        # Check that the same args always return the same color.
        assert picker.pick(*args).hex == color.hex
        assert picker.pick(*args).hex == color.hex
