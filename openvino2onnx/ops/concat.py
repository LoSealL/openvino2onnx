"""
Copyright Wenyi Tang 2023

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

"""

from onnx import AttributeProto

from . import Node, register


@register
class Concat(Node):
    @property
    def type_name(self):
        return "Concat"

    @property
    def attributes(self):
        axis = AttributeProto(name="axis", type=AttributeProto.INT, i=int(self.axis))
        return (axis,)
