"""
Copyright Wenyi Tang 2023

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

"""

from . import Node, register


@register
class Clamp(Node):
    """https://onnx.ai/onnx/operators/onnx__Clip.html"""

    @property
    def type_name(self) -> str:
        return "Clip"


@register
class PReLU(Node):
    """https://onnx.ai/onnx/operators/onnx__PRelu.html"""

    @property
    def type_name(self) -> str:
        return "PRelu"


@register
class ReLU(Node):
    """https://onnx.ai/onnx/operators/onnx__Relu.html"""

    @property
    def type_name(self):
        return "Relu"


@register
class SoftMax(Node):
    """https://onnx.ai/onnx/operators/onnx__Softmax.html"""

    @property
    def type_name(self) -> str:
        return "Softmax"
