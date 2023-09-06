"""
Copyright Wenyi Tang 2023

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

"""

from typing import Iterator

from onnx import AttributeProto

from . import Node, register


@register
class VariadicSplit(Node):
    @property
    def type_name(self) -> str:
        return "Split"

    @property
    def attributes(self) -> Iterator[AttributeProto]:
        axis = AttributeProto(name="axis", type=AttributeProto.INT, i=int(self.axis))
        return (axis,)
