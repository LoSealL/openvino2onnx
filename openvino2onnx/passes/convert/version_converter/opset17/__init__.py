"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from openvino2onnx.passes.convert.version_converter import (
    OP_CONVERTER_17 as OP_CONVERTER,
)

__all__ = ["OP_CONVERTER"]
"""
Downgrade ai.onnx opset 19 to opset 17.
"""
