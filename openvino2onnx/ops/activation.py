"""
Copyright Wenyi Tang 2023

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

"""

from . import Node, register


@register
class ReLU(Node):
    @property
    def type_name(self):
        return "Relu"


@register
class SoftMax(Node):
    @property
    def type_name(self) -> str:
        return "Softmax"
