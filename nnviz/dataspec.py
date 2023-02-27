from __future__ import annotations
from abc import ABC, abstractmethod

import typing as t

import pydantic as pyd
import torch


class DataSpecVisitor(ABC):
    """Visitor for the `DataSpec` class."""

    @property
    def has_control(self) -> bool:
        """Returns whether the visitor has control over the traversal."""
        return False

    @abstractmethod
    def visit_tensor_spec(self, spec: TensorSpec) -> None:
        """Visits a `TensorSpec`."""
        pass

    @abstractmethod
    def visit_builtin_spec(self, spec: BuiltInSpec) -> None:
        """Visits a `BuiltInSpec`."""
        pass

    @abstractmethod
    def visit_list_spec(self, spec: ListSpec) -> None:
        """Visits a `ListSpec`."""
        pass

    @abstractmethod
    def visit_map_spec(self, spec: MapSpec) -> None:
        """Visits a `MapSpec`."""
        pass

    @abstractmethod
    def visit_unknown_spec(self, spec: UnknownSpec) -> None:
        """Visits a `UnknownSpec`."""
        pass


class PrettyDataSpecVisitor(DataSpecVisitor):
    """Visitor for the `DataSpec` class that produces a pretty string representation."""

    def __init__(self):
        self._indent = 4
        self._result = ""

        self._template = "{entry}\n"

    @property
    def has_control(self) -> bool:
        return True

    @property
    def result(self) -> str:
        return self._result

    def visit_tensor_spec(self, spec: TensorSpec) -> None:
        entry = f"Tensor(shape={spec.shape}, dtype={spec.dtype})"
        self._result += self._template.format(entry=entry)

    def visit_builtin_spec(self, spec: BuiltInSpec) -> None:
        self._result += self._template.format(entry=spec.name)

    def visit_list_spec(self, spec: ListSpec) -> None:
        self._result += self._template.format(entry="List: [")
        for i, element in enumerate(spec.elements):
            old_template = self._template
            self._template = " " * self._indent + f"{i}: {self._template}"
            element.accept(self)
            self._template = old_template
        self._result += self._template.format(entry="]")

    def visit_map_spec(self, spec: MapSpec) -> None:
        self._result += self._template.format(entry="Map: {")
        for k, element in spec.elements.items():
            old_template = self._template
            self._template = " " * self._indent + f"{k}: {self._template}"
            element.accept(self)
            self._template = old_template
        self._result += self._template.format(entry="}")

    def visit_unknown_spec(self, spec: UnknownSpec) -> None:
        self._result += self._template.format(entry="???")


class DataSpec(pyd.BaseModel, ABC):
    """Specification of the data that is passed through the graph."""

    class Config:
        frozen = True

    spec_type: t.Literal[""] = ""

    @abstractmethod
    def accept(self, visitor: DataSpecVisitor) -> None:
        """Accepts an incoming visitor."""
        pass

    @classmethod
    def build(cls, data: t.Any) -> DataSpec:
        if isinstance(data, torch.Tensor):
            return TensorSpec(shape=data.shape, dtype=str(data.dtype))
        elif isinstance(data, (list, tuple)):
            return ListSpec(elements=[cls.build(x) for x in data])
        elif isinstance(data, dict):
            return MapSpec(elements={k: cls.build(v) for k, v in data.items()})
        elif isinstance(data, (int, float, str, bool, type(None), type(...), bytes)):
            return BuiltInSpec(name=data.__class__.__name__)
        else:
            return UnknownSpec()

    def pretty(self) -> str:
        visitor = PrettyDataSpecVisitor()
        self.accept(visitor)
        return visitor.result


class TensorSpec(DataSpec):
    """Specification of the data that is passed through the graph."""

    spec_type: t.Literal["tensor"] = "tensor"
    shape: t.Sequence[int] = pyd.Field(
        default_factory=list, description="Shape of the tensor."
    )
    dtype: str = pyd.Field("", description="Data type of the tensor.")

    def accept(self, visitor: DataSpecVisitor) -> None:
        visitor.visit_tensor_spec(self)


class BuiltInSpec(DataSpec):
    """Specification of the data that is passed through the graph."""

    spec_type: t.Literal["builtin"] = "builtin"
    name: str = pyd.Field("", description="Name of the builtin type.")

    def accept(self, visitor: DataSpecVisitor) -> None:
        visitor.visit_builtin_spec(self)


class UnknownSpec(DataSpec):
    """Specification of the data that is passed through the graph."""

    spec_type: t.Literal["unknown"] = "unknown"

    def accept(self, visitor: DataSpecVisitor) -> None:
        visitor.visit_unknown_spec(self)


class ListSpec(DataSpec):
    """Specification of the data that is passed through the graph."""

    spec_type: t.Literal["list"] = "list"

    elements: t.Sequence[DataSpec] = pyd.Field(
        default_factory=list, description="List of elements in the list."
    )

    @pyd.validator("elements")
    def validate_elements(cls, v):
        return [pyd.parse_obj_as(t_any_spec, v) for v in v]

    def accept(self, visitor: DataSpecVisitor) -> None:
        visitor.visit_list_spec(self)
        if not visitor.has_control:
            for element in self.elements:
                element.accept(visitor)


class MapSpec(DataSpec):
    """Specification of the data that is passed through the graph."""

    spec_type: t.Literal["map"] = "map"

    elements: t.Mapping[str, DataSpec] = pyd.Field(
        default_factory=dict, description="Mapping of keys to elements in the map."
    )

    @pyd.validator("elements")
    def validate_elements(cls, v):
        return {k: pyd.parse_obj_as(t_any_spec, v) for k, v in v.items()}

    def accept(self, visitor: DataSpecVisitor) -> None:
        visitor.visit_map_spec(self)
        if not visitor.has_control:
            for element in self.elements.values():
                element.accept(visitor)


t_any_spec = t.Union[TensorSpec, BuiltInSpec, ListSpec, MapSpec]
