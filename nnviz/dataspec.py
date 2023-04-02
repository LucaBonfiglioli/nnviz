from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod

import pydantic as pyd
import torch


class DataSpecVisitor(ABC):
    """Visitor for the `DataSpec` class."""

    @property
    def has_control(self) -> bool:  # pragma: no cover
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
        """Constructor. Accepts no arguments and returns an initialized visitor."""
        self._indent = 0
        self._key = ""
        self._template = "{indent}{key}{entry}\n"

        self._result = ""

    @property
    def has_control(self) -> bool:
        return True

    @property
    def result(self) -> str:
        """The result of the visitation. This property is only valid after the
        `visit` method has been called, or after the `accept` method has been
        called on a `DataSpec` instance, passin this visitor as an argument.

        Returns:
            str: A string representation of the visited `DataSpec` instance.
        """
        return self._result

    def _entry(self, body: str) -> str:
        key_fmt = self._key + ": " if self._key else ""
        indent = " " * 4 * self._indent
        return self._template.format(indent=indent, key=key_fmt, entry=body)

    def visit_tensor_spec(self, spec: TensorSpec) -> None:
        entry = f"Tensor(shape={spec.shape}, dtype={spec.dtype})"
        self._result += self._entry(entry)

    def visit_builtin_spec(self, spec: BuiltInSpec) -> None:
        self._result += self._entry(spec.name)

    def _begin_composite(self, line: str) -> str:
        self._result += self._entry(line)
        self._indent += 1
        return self._key

    def _end_composite(self, line: str, old_key: str) -> None:
        self._key = ""
        self._indent -= 1
        self._result += self._entry(line)
        self._key = old_key

    def visit_list_spec(self, spec: ListSpec) -> None:
        _old_key = self._begin_composite("List: [")
        for i, element in enumerate(spec.elements):
            self._key = str(i)
            element.accept(self)
        self._end_composite("]", _old_key)

    def visit_map_spec(self, spec: MapSpec) -> None:
        _old_key = self._begin_composite("Map: {")
        for k, element in spec.elements.items():
            self._key = k
            element.accept(self)
        self._end_composite("}", _old_key)

    def visit_unknown_spec(self, spec: UnknownSpec) -> None:
        self._result += self._entry("???")


class DataSpec(pyd.BaseModel, ABC):
    """Models a synthetic representation of the data type passing through the graph.
    Currently, this hierarchy can model tensors, builtin types, lists and maps. All
    other types are represented as `UnknownSpec`.
    """

    class Config:
        frozen = True

    spec_type: t.Literal[""] = ""
    """DataSpec type discriminator. Do not set this field manually."""

    @abstractmethod
    def accept(self, visitor: DataSpecVisitor) -> None:
        """Accepts an incoming visitor."""
        pass

    @classmethod
    def build(cls, data: t.Any) -> DataSpec:
        """Creates a `DataSpec` instance from a given data object.

        Args:
            data (t.Any): The data object to build the `DataSpec` from. Can be anything.

        Returns:
            DataSpec: The `DataSpec` instance that models the given data object.
        """
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
        """Pretty string representation for terminal output.

        Returns:
            str: A pretty string representation of the `DataSpec` instance.
        """
        visitor = PrettyDataSpecVisitor()
        self.accept(visitor)
        return visitor.result


class TensorSpec(DataSpec):
    """Specification of the data that is passed through the graph."""

    spec_type: t.Literal["tensor"] = "tensor"
    """DataSpec type discriminator. Do not set this field manually."""

    shape: t.Sequence[int] = pyd.Field(default_factory=list)
    """Shape of the tensor."""

    dtype: str = ""
    """Data type of the tensor."""

    def accept(self, visitor: DataSpecVisitor) -> None:
        visitor.visit_tensor_spec(self)


class BuiltInSpec(DataSpec):
    """Specification of the data that is passed through the graph."""

    spec_type: t.Literal["builtin"] = "builtin"
    """DataSpec type discriminator. Do not set this field manually."""

    name: str = ""
    """Name of the builtin type."""

    def accept(self, visitor: DataSpecVisitor) -> None:
        visitor.visit_builtin_spec(self)


class UnknownSpec(DataSpec):
    """Specification of the data that is passed through the graph."""

    spec_type: t.Literal["unknown"] = "unknown"
    """DataSpec type discriminator. Do not set this field manually."""

    def accept(self, visitor: DataSpecVisitor) -> None:
        visitor.visit_unknown_spec(self)


class ListSpec(DataSpec):
    """Specification of the data that is passed through the graph."""

    spec_type: t.Literal["list"] = "list"
    """DataSpec type discriminator. Do not set this field manually."""

    elements: t.Sequence[t_any_spec] = pyd.Field(default_factory=list)
    """List of elements in the list."""

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
    """DataSpec type discriminator. Do not set this field manually."""

    elements: t.Mapping[str, t_any_spec] = pyd.Field(default_factory=dict)
    """Mapping of keys to elements in the map."""

    @pyd.validator("elements")
    def validate_elements(cls, v):
        return {k: pyd.parse_obj_as(t_any_spec, v) for k, v in v.items()}

    def accept(self, visitor: DataSpecVisitor) -> None:
        visitor.visit_map_spec(self)
        if not visitor.has_control:
            for element in self.elements.values():
                element.accept(visitor)


t_any_spec = t.Union[DataSpec, TensorSpec, BuiltInSpec, ListSpec, MapSpec]

ListSpec.update_forward_refs()
MapSpec.update_forward_refs()
