# This code is cursed, don't try to understand it, just run it and see what happens.

import colorsys
import math
import random
import typing as t

import cv2 as cv
import numpy as np

from nnviz.colors.bubble import _BubbleTree

if __name__ == "__main__":
    H = W = 512
    winname = "Bubble Color Picker"
    cv.namedWindow(winname, cv.WINDOW_NORMAL)
    cv.resizeWindow(winname, H, W)
    bubble = _BubbleTree(np.array([0, 0]), 1.0, {})
    depth = 3
    possible_keys = [chr(i) for i in range(ord("a"), ord("a") + 21)]
    while True:
        canvas = np.zeros((H, W, 3))
        args = [random.choice(possible_keys) for _ in range(depth)]
        args += [random.random() for _ in range(2)]

        def add_args(tree: _BubbleTree, args: t.List[t.Hashable]) -> None:
            if len(args) == 0:
                return
            if args[0] not in tree:
                tree.spawn(args[0])
            add_args(tree[args[0]], args[1:])

        add_args(bubble, args)  # type: ignore

        def visit(tree: _BubbleTree, d: int) -> None:
            row = int(tree.origin[0] * H // 2 + H // 2)
            col = int(tree.origin[1] * H // 2 + H // 2)
            if len(tree.children) == 0:
                canvas[row, col, :] = 255
            if d <= depth and d > 0:
                hue = math.atan2(*tree.origin) / (2 * math.pi) + 0.5
                sat = math.sqrt(tree.origin[0] ** 2 + tree.origin[1] ** 2)
                color = colorsys.hsv_to_rgb(hue, sat, 1.0)
                color = int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
                cv.circle(
                    canvas,
                    (col, row),
                    int(tree.radius * H // 2),
                    color=color,
                    thickness=4 // d,
                )
            for child in tree.children.values():
                visit(child, d + 1)

        visit(bubble, 0)
        canvas = (canvas / np.max(canvas) * 255).astype(np.uint8)
        cv.imshow(winname, cv.cvtColor(canvas, cv.COLOR_RGB2BGR))
        key = cv.waitKey(1)
        if key == ord("q"):
            break
