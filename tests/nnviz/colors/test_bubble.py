import typing as t

import pytest

from nnviz import colors


class TestBubbleColorPicker:
    @pytest.mark.parametrize(
        "seq_of_args",
        [
            [(1, 2, 3), ("a", "b", "c"), (1, "a", 2, "b"), (1, 2, 3, 10)],
            [("a", "b", "c"), (1, "a", 2, "b"), (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)],
            [(1, "a", 2, "b"), ("a", "b", "c"), (1, 2, 3, 4, 5, "f")],
            [(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), (1, 2, 3), ("a", "b", "c")],
            [tuple(), tuple()],
            [(1, 2, 3), ("a", "b", "c"), tuple(), (1, "a", 2, "b")],
        ],
    )
    def test_pick(self, seq_of_args: t.Sequence[t.Sequence[t.Hashable]]):
        picker = colors.BubbleColorPicker()
        for args in seq_of_args:
            color = picker.pick(*args)
            assert isinstance(color, colors.RGBColor)

            # Check that the same args always return the same color.
            assert picker.pick(*args).to_hex() == color.to_hex()
