"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

"""

from . import Node, register


@register
class HSwish(Node):
    """https://onnx.ai/onnx/operators/onnx__HardSwish.html"""

    @property
    def type_name(self) -> str:
        return "HardSwish"
