"""
Copyright Wenyi Tang 2023

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

"""

from onnx import AttributeProto

from . import Node, register


@register
class Transpose(Node):
    @property
    def type_name(self):
        return "Transpose"

    @property
    def attributes(self):
        if len(self.inputs) == 1:
            perm = AttributeProto(
                name="perm",
                type=AttributeProto.INTS,
                ints=map(int, self.perm.split(",")),
            )
            return (perm,)
        return []
