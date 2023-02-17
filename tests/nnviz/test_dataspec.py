import typing as t

import pytest
import torch

from nnviz import dataspec as ds


class MockVisitor(ds.DataSpecVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.visited_tensors: t.List[ds.TensorSpec] = []
        self.visited_lists: t.List[ds.ListSpec] = []
        self.visited_maps: t.List[ds.MapSpec] = []
        self.visited_builtins: t.List[ds.BuiltInSpec] = []
        self.visited_unknowns: t.List[ds.UnknownSpec] = []

    def visit_tensor_spec(self, spec: ds.TensorSpec) -> None:
        self.visited_tensors.append(spec)

    def visit_list_spec(self, spec: ds.ListSpec) -> None:
        self.visited_lists.append(spec)

    def visit_map_spec(self, spec: ds.MapSpec) -> None:
        self.visited_maps.append(spec)

    def visit_builtin_spec(self, spec: ds.BuiltInSpec) -> None:
        self.visited_builtins.append(spec)

    def visit_unknown_spec(self, spec: ds.UnknownSpec) -> None:
        self.visited_unknowns.append(spec)


class TestTensorSpec:
    def test_accept(self) -> None:
        spec = ds.TensorSpec(shape=(1, 3, 224, 224), dtype="torch.float32")
        visitor = MockVisitor()
        spec.accept(visitor)
        assert visitor.visited_tensors == [spec]


class TestListSpec:
    def test_accept(self) -> None:
        spec = ds.ListSpec(
            elements=[
                ds.TensorSpec(shape=(1, 3), dtype="torch.float32"),
                ds.TensorSpec(shape=(4, 3), dtype="torch.float32"),
                ds.ListSpec(
                    elements=[
                        ds.TensorSpec(shape=(6, 4), dtype="torch.float32"),
                        ds.TensorSpec(shape=(4,), dtype="torch.float32"),
                    ]
                ),
            ],
        )
        visitor = MockVisitor()
        spec.accept(visitor)
        assert visitor.visited_lists == [spec, spec.elements[2]]
        assert visitor.visited_tensors == [
            spec.elements[0],
            spec.elements[1],
            spec.elements[2].elements[0],  # type: ignore
            spec.elements[2].elements[1],  # type: ignore
        ]


class TestBuiltInSpec:
    def test_accept(self) -> None:
        spec = ds.BuiltInSpec(name="int")
        visitor = MockVisitor()
        spec.accept(visitor)
        assert visitor.visited_builtins == [spec]


class TestUnknownSpec:
    def test_accept(self) -> None:
        spec = ds.UnknownSpec()
        visitor = MockVisitor()
        spec.accept(visitor)
        assert visitor.visited_unknowns == [spec]


class TestMapSpec:
    def test_accept(self) -> None:
        spec = ds.MapSpec(
            elements={
                "a": ds.TensorSpec(shape=(1, 3), dtype="torch.float32"),
                "b": ds.TensorSpec(shape=(4, 3), dtype="torch.float32"),
                "c": ds.ListSpec(
                    elements=[
                        ds.TensorSpec(shape=(6, 4), dtype="torch.float32"),
                        ds.TensorSpec(shape=(4,), dtype="torch.float32"),
                    ]
                ),
            },
        )
        visitor = MockVisitor()
        spec.accept(visitor)
        assert visitor.visited_maps == [spec]
        assert visitor.visited_lists == [spec.elements["c"]]
        assert visitor.visited_tensors == [
            spec.elements["a"],
            spec.elements["b"],
            spec.elements["c"].elements[0],  # type: ignore
            spec.elements["c"].elements[1],  # type: ignore
        ]


class TestDataSpec:
    @pytest.mark.parametrize(
        ["data", "spec"],
        [
            (
                torch.zeros(1, 3, 224, 224),
                ds.TensorSpec(shape=(1, 3, 224, 224), dtype="torch.float32"),
            ),
            (
                torch.zeros(4, 3, 16, 320, dtype=torch.int16),
                ds.TensorSpec(shape=(4, 3, 16, 320), dtype="torch.int16"),
            ),
            (
                [
                    torch.zeros(1, 3),
                    torch.zeros(4, 3),
                    (torch.zeros(6, 4), torch.zeros(4)),
                ],
                ds.ListSpec(
                    elements=[
                        ds.TensorSpec(shape=(1, 3), dtype="torch.float32"),
                        ds.TensorSpec(shape=(4, 3), dtype="torch.float32"),
                        ds.ListSpec(
                            elements=[
                                ds.TensorSpec(shape=(6, 4), dtype="torch.float32"),
                                ds.TensorSpec(shape=(4,), dtype="torch.float32"),
                            ]
                        ),
                    ],
                ),
            ),
            (
                {
                    "a": torch.zeros(1, 3),
                    "b": torch.zeros(4, 3),
                    "c": (torch.zeros(6, 4), torch.zeros(4)),
                },
                ds.MapSpec(
                    elements={
                        "a": ds.TensorSpec(shape=(1, 3), dtype="torch.float32"),
                        "b": ds.TensorSpec(shape=(4, 3), dtype="torch.float32"),
                        "c": ds.ListSpec(
                            elements=[
                                ds.TensorSpec(shape=(6, 4), dtype="torch.float32"),
                                ds.TensorSpec(shape=(4,), dtype="torch.float32"),
                            ]
                        ),
                    },
                ),
            ),
            (16, ds.BuiltInSpec(name="int")),
            (16.0, ds.BuiltInSpec(name="float")),
            (True, ds.BuiltInSpec(name="bool")),
            (None, ds.BuiltInSpec(name="NoneType")),
            ("hello", ds.BuiltInSpec(name="str")),
            (b"hello", ds.BuiltInSpec(name="bytes")),
            (object(), ds.UnknownSpec()),
        ],
    )
    def test_build(self, data: t.Any, spec: ds.DataSpec) -> None:
        assert ds.DataSpec.build(data) == spec
