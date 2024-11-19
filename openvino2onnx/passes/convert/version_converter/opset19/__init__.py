"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from openvino2onnx.passes.convert.version_converter import (
    OP_CONVERTER_19 as OP_CONVERTER,
)

__all__ = ["OP_CONVERTER"]
"""
Downgrade ai.onnx opset 22 to opset 19.
"""
