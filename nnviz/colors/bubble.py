from __future__ import annotations

import typing as t

import numpy as np

from nnviz import colors


class _BubbleTree:
    def __init__(
        self,
        origin: np.ndarray,
        radius: float,
        children: t.Dict[t.Hashable, _BubbleTree],
    ) -> None:
        self.origin = origin.astype(np.float64)
        self.radius = float(radius)
        self.children = children

        # TODO: Make this configurable
        self._factor = 0.3
        self._ord = 2

    def __getitem__(self, key: t.Hashable) -> _BubbleTree:
        return self.children[key]

    def __setitem__(self, key: t.Hashable, value: _BubbleTree) -> None:
        self.children[key] = value

    def __contains__(self, key: t.Hashable) -> bool:
        return key in self.children

    def spawn(self, key: t.Hashable) -> None:
        # If there are no children, we can just return a new subtree with the origin
        # as the origin of the new subtree but smaller radius
        if len(self.children) == 0:
            spawned = _BubbleTree(self.origin, self.radius * self._factor, {})
            self.children[key] = spawned
            return

        # Start from the origin of the parent tree
        children_origins = np.stack([x.origin for x in self.children.values()])
        children_radii = np.stack([x.radius for x in self.children.values()])
        eps = self.radius * 1e-5

        # Compute the pairwise distances between the children
        distances = np.linalg.norm(
            children_origins[:, None, :] - children_origins[None, :, :],
            ord=self._ord,
            axis=2,
        )
        distances -= children_radii[:, None] + children_radii[None, :]

        # Select the child with the largest distance to the closest child
        starting_child_index = np.argmax(
            np.min(distances + np.eye(distances.shape[0]) * self.radius, axis=1)
        )
        random_child_org = children_origins[starting_child_index]
        random_child_rad = children_radii[starting_child_index]

        # Select a random direction
        direction = np.random.uniform(-1, 1, size=children_origins.shape[1])
        # Normalize the direction so that it has length children radius
        direction = (
            direction / np.linalg.norm(direction, ord=self._ord) * random_child_rad
        )

        # Place the new subtree at the origin of the random child plus the direction
        new_origin = random_child_org + direction

        budget = 1000
        while budget > 0:
            # Compute the distance to the edge of the parent tree
            distance_to_edge = self.radius - np.linalg.norm(
                new_origin - self.origin, ord=self._ord  # type: ignore
            )

            # Compute the distance to the closest child
            distances = np.linalg.norm(
                new_origin - children_origins, ord=self._ord, axis=1  # type: ignore
            )

            # We have to subtract the radius of the child from the distance to the child
            # because the distance is computed from the center of the child
            distances -= children_radii

            # Select the closest child
            min_index, min_distance = np.argmin(distances), np.min(distances)

            # If the distance to the edge is smaller than the half distance to the closest
            # child, the direction is towards the center of the parent tree
            if distance_to_edge < min_distance:
                direction = self.origin - new_origin

            # If the distance to the edge is larger than the distance to the closest
            # child, the direction is away from the center of the closest child
            else:
                direction = new_origin - children_origins[min_index]

            # Normalize the direction
            direction /= np.linalg.norm(direction, ord=self._ord) + eps  # type: ignore

            # Move the new origin in the direction of the closest child
            new_origin += direction * self.radius * 0.01

            # Reduce the budget
            budget -= 1

        # The new radius is the distance to the closest child
        new_radius = min(min_distance, distance_to_edge)  # type: ignore

        # Clamp the raidus to [eps, self.radius * self._factor]
        new_radius = max(eps, min(new_radius, self.radius * self._factor))

        spawned = _BubbleTree(new_origin, new_radius, {})
        self.children[key] = spawned


class BubbleColorPicker(colors.ColorPicker):
    def __init__(self) -> None:
        super().__init__()

        self._color_tree = _BubbleTree(np.array([127, 127, 127]), 127, {})
        self._color_tree.spawn("__root__")
        self._factor = 0.5
        self._ord = "fro"

    def _np_to_rgb(self, np_color: np.ndarray) -> colors.RGBColor:
        return colors.RGBColor(*np_color.astype(np.uint8).tolist())

    def _pick_recur(self, sub_tree: _BubbleTree, *args: t.Hashable) -> colors.RGBColor:
        # If there are no arguments left, we have reached the end of the path and we can
        # return the color at the end of the path
        if len(args) == 0:
            return self._np_to_rgb(sub_tree.origin)

        # If the first argument is not in the sub-tree, we need to select a new color
        # close to the origin color and add it to the sub-tree
        if args[0] not in sub_tree:
            sub_tree.spawn(args[0])

        # Recurse on the sub-tree with the first argument removed
        return self._pick_recur(sub_tree[args[0]], *args[1:])

    def pick(self, *args: t.Hashable) -> colors.RGBColor:
        return self._pick_recur(self._color_tree, *args)


if __name__ == "__main__":
    import cv2 as cv
    import random

    cv.namedWindow("sasso", cv.WINDOW_NORMAL)
    H, W = 256, 256
    cv.resizeWindow("sasso", H, W)

    bubble = _BubbleTree(np.array([0, 0]), 1.0, {})

    depth = 3
    possible_keys = ["a", "b", "c", "d", "e", "f", "g"]

    while True:

        canvas = np.zeros((H, W, 3))

        # Sample with replacement from the possible keys
        args = [random.choice(possible_keys) for _ in range(depth)]
        # Add some other random arguments
        args += [random.random() for _ in range(2)]

        def add_args(tree: _BubbleTree, args: t.List[t.Hashable]) -> None:
            if len(args) == 0:
                return

            if args[0] not in tree:
                tree.spawn(args[0])

            add_args(tree[args[0]], args[1:])

        add_args(bubble, args)  # type: ignore

        def visit(tree: _BubbleTree, d: int) -> None:
            row, col = int(tree.origin[0] * H // 2 + H // 2), int(
                tree.origin[1] * H // 2 + H // 2
            )
            if len(tree.children) == 0:
                canvas[row, col, :] = 255

            if d <= depth and d > 0:
                # Draw a circle around the origin of the tree
                cv.circle(
                    canvas,
                    (col, row),
                    int(tree.radius * H // 2),
                    (0, 0, 255 // d),
                    thickness=1,
                )

            for child in tree.children.values():
                visit(child, d + 1)

        visit(bubble, 0)

        print(".".join(args[:depth]))  # type: ignore
        canvas = (canvas / np.max(canvas) * 255).astype(np.uint8)
        cv.imshow("sasso", cv.cvtColor(canvas, cv.COLOR_RGB2BGR))
        key = cv.waitKey(0)

        if key == ord("q"):
            break
