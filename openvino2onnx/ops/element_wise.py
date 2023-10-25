"""
Copyright Wenyi Tang 2023

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

"""

from . import Node, register


@register
class Subtract(Node):
    """https://onnx.ai/onnx/operators/onnx__Sub.html"""

    @property
    def type_name(self):
        return "Sub"


@register
class Multiply(Node):
    """https://onnx.ai/onnx/operators/onnx__Mul.html"""

    @property
    def type_name(self):
        return "Mul"


@register
class Divide(Node):
    """https://onnx.ai/onnx/operators/onnx__Div.html"""

    @property
    def type_name(self) -> str:
        return "Div"
