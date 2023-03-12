import typing as t

import pytest
import torch

from nnviz import dataspec as ds


class MockVisitor(ds.DataSpecVisitor):
    def __init__(self, has_control: bool) -> None:
        super().__init__()
        self._has_control = has_control
        self.visited_tensors: t.List[ds.TensorSpec] = []
        self.visited_lists: t.List[ds.ListSpec] = []
        self.visited_maps: t.List[ds.MapSpec] = []
        self.visited_builtins: t.List[ds.BuiltInSpec] = []
        self.visited_unknowns: t.List[ds.UnknownSpec] = []

    @property
    def has_control(self) -> bool:
        return self._has_control

    def visit_tensor_spec(self, spec: ds.TensorSpec) -> None:
        self.visited_tensors.append(spec)

    def visit_list_spec(self, spec: ds.ListSpec) -> None:
        self.visited_lists.append(spec)
        if self.has_control:
            for element in spec.elements:
                element.accept(self)

    def visit_map_spec(self, spec: ds.MapSpec) -> None:
        self.visited_maps.append(spec)
        if self.has_control:
            for element in spec.elements.values():
                element.accept(self)

    def visit_builtin_spec(self, spec: ds.BuiltInSpec) -> None:
        self.visited_builtins.append(spec)

    def visit_unknown_spec(self, spec: ds.UnknownSpec) -> None:
        self.visited_unknowns.append(spec)


@pytest.fixture(params=[True, False])
def mock_visitor(request) -> MockVisitor:
    return MockVisitor(request.param)


class TestTensorSpec:
    def test_accept(self, mock_visitor: MockVisitor) -> None:
        spec = ds.TensorSpec(shape=(1, 3, 224, 224), dtype="torch.float32")
        spec.accept(mock_visitor)
        assert mock_visitor.visited_tensors == [spec]


class TestListSpec:
    def test_accept(self, mock_visitor: MockVisitor) -> None:
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
        spec.accept(mock_visitor)
        assert mock_visitor.visited_lists == [spec, spec.elements[2]]
        assert mock_visitor.visited_tensors == [
            spec.elements[0],
            spec.elements[1],
            spec.elements[2].elements[0],  # type: ignore
            spec.elements[2].elements[1],  # type: ignore
        ]


class TestBuiltInSpec:
    def test_accept(self, mock_visitor: MockVisitor) -> None:
        spec = ds.BuiltInSpec(name="int")
        spec.accept(mock_visitor)
        assert mock_visitor.visited_builtins == [spec]


class TestUnknownSpec:
    def test_accept(self, mock_visitor: MockVisitor) -> None:
        spec = ds.UnknownSpec()
        spec.accept(mock_visitor)
        assert mock_visitor.visited_unknowns == [spec]


class TestMapSpec:
    def test_accept(self, mock_visitor: MockVisitor) -> None:
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
        spec.accept(mock_visitor)
        assert mock_visitor.visited_maps == [spec]
        assert mock_visitor.visited_lists == [spec.elements["c"]]
        assert mock_visitor.visited_tensors == [
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


class TestPrettyDataSpecVisitor:
    def test_has_control(self) -> None:
        assert ds.PrettyDataSpecVisitor().has_control

    def test_result(self) -> None:
        assert isinstance(ds.PrettyDataSpecVisitor().result, str)

    @pytest.mark.parametrize(
        "spec",
        [
            ds.TensorSpec(shape=(1, 3), dtype="torch.float32"),
            ds.TensorSpec(shape=(4, 3), dtype="torch.uint8"),
            ds.TensorSpec(shape=tuple(), dtype="torch.int16"),
        ],
    )
    def test_visit_tensor_spec(self, spec: ds.TensorSpec) -> None:
        visitor = ds.PrettyDataSpecVisitor()
        visitor.visit_tensor_spec(spec)
        must_be_present = [
            "Tensor",
            "shape",
            "dtype",
            *[str(x) for x in spec.shape],
            spec.dtype,
        ]
        for s in must_be_present:
            assert s in visitor.result

        assert spec.pretty() == visitor.result

    @pytest.mark.parametrize("spec", [ds.UnknownSpec()])
    def test_visit_unknown_spec(self, spec: ds.UnknownSpec) -> None:
        visitor = ds.PrettyDataSpecVisitor()
        visitor.visit_unknown_spec(spec)
        # Don't care about the exact string, just that it's not empty
        assert len(visitor.result) > 0

        assert spec.pretty() == visitor.result

    @pytest.mark.parametrize(
        "spec",
        [
            ds.BuiltInSpec(name="int"),
            ds.BuiltInSpec(name="float"),
            ds.BuiltInSpec(name="bool"),
            ds.BuiltInSpec(name="NoneType"),
        ],
    )
    def test_visit_builtin_spec(self, spec: ds.BuiltInSpec) -> None:
        visitor = ds.PrettyDataSpecVisitor()
        visitor.visit_builtin_spec(spec)
        assert spec.name in visitor.result

        assert spec.pretty() == visitor.result

    @pytest.mark.parametrize(
        "spec",
        [
            ds.ListSpec(elements=[]),
            ds.ListSpec(
                elements=[
                    ds.TensorSpec(shape=(1, 3), dtype="torch.float32"),
                    ds.TensorSpec(shape=(4, 3), dtype="torch.float32"),
                ]
            ),
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
                ]
            ),
        ],
    )
    def test_visit_list_spec(self, spec: ds.ListSpec) -> None:
        visitor = ds.PrettyDataSpecVisitor()
        visitor.visit_list_spec(spec)
        print(visitor.result)
        lines = visitor.result.splitlines()
        assert len(lines) >= 2
        assert "[" in lines[0]
        assert "]" in lines[-1]
        body_lines = lines[1:-1]
        assert len(body_lines) >= len(spec.elements)

        for i in range(len(spec.elements)):
            assert str(i) in visitor.result

        for sub_spec in spec.elements:
            sub_visitor = ds.PrettyDataSpecVisitor()
            sub_spec.accept(sub_visitor)
            spec_lines = sub_visitor.result.splitlines()
            for line in spec_lines:
                assert line.strip() in visitor.result

        assert spec.pretty() == visitor.result

    @pytest.mark.parametrize(
        "spec",
        [
            ds.MapSpec(elements={}),
            ds.MapSpec(
                elements={
                    "a": ds.TensorSpec(shape=(1, 3), dtype="torch.float32"),
                    "b": ds.TensorSpec(shape=(4, 3), dtype="torch.float32"),
                }
            ),
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
                }
            ),
        ],
    )
    def test_visit_map_spec(self, spec: ds.MapSpec) -> None:
        visitor = ds.PrettyDataSpecVisitor()
        visitor.visit_map_spec(spec)
        print(visitor.result)
        lines = visitor.result.splitlines()
        assert len(lines) >= 2
        assert "{" in lines[0]
        assert "}" in lines[-1]
        body_lines = lines[1:-1]
        assert len(body_lines) >= len(spec.elements)

        assert spec.pretty() == visitor.result
