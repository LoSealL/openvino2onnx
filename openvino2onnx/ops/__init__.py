"""
Copyright Wenyi Tang 2023

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

Mapping openvino opset to onnx opset
"""

import importlib
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Iterator, Tuple

from onnx import AttributeProto


class Node(metaclass=ABCMeta):
    """A base node object that exposes two interfaces.

    - type_name to specify the onnx type name of this node;
    - attributes to provide onnx attributes of the node.
    """

    def __init__(self, **attrs) -> None:
        self.__dict__.update(attrs)

    @property
    @abstractmethod
    def type_name(self) -> str:
        """Subclass must inherit this property and return a proper name,
        the name must be in the op list of onnx document.
        """
        return ""

    @property
    def attributes(self) -> Iterator[AttributeProto]:
        """Return a list of attributes for this node."""
        return []

    def __contains__(self, item: str) -> bool:
        return item in self.__dict__


class Registry:
    """A registry to collect all op mappings"""

    _reg = {}

    @classmethod
    def get(cls, name: str):
        """Get the registered node object."""
        return cls._reg.get(name)

    @classmethod
    def register(cls, name, obj, *, allow_override=False):
        """Register a new node type."""
        if name in cls._reg and not allow_override:
            raise KeyError(f"{name} has already been registered!")
        cls._reg[name] = obj


def register(func):
    """A decorator to register node subclass.

    Example::

        @register
        class Convolution(Node):
            @property
            def type_name(self):
                return "Conv"
    """
    Registry.register(func.__name__, func)
    return func


_LOADED = False


def auto_load():
    """Lazily load modules in ops folder"""
    for file in Path(__file__).parent.rglob("*.py"):
        if file.stem == "__init__":
            continue
        importlib.import_module("." + file.stem, "openvino2onnx.ops")
    return True


def get_onnx_optype_and_attributes(
    ir_node: dict,
) -> Tuple[str, Iterator[AttributeProto]]:
    """Get onnx type name and node attributes

    Args:
        ir_node (dict): a node in the graph.

    Returns:
        Tuple[str, Iterator[AttributeProto]]: a tuple of type name and attributes.
    """
    global _LOADED  # pylint: disable=global-statement
    if not _LOADED:
        _LOADED = auto_load()

    if builder := Registry.get(ir_node["type"]):
        node = builder(**ir_node)
        return node.type_name, node.attributes
    return ir_node["type"], []
