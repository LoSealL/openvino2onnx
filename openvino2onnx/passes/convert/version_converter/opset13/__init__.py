"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from openvino2onnx.passes.convert.version_converter import (
    OP_CONVERTER_13 as OP_CONVERTER,
)

__all__ = ["OP_CONVERTER"]
"""
Downgrade ai.onnx opset 17 to opset 13.
"""
