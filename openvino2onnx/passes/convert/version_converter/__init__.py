"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from openvino2onnx.passes import Registry

OP_CONVERTER_19 = Registry("op_converter_19")
OP_CONVERTER_17 = Registry("op_converter_17")
OP_CONVERTER_13 = Registry("op_converter_13")
OP_CONVERTER = {
    13: OP_CONVERTER_13,
    17: OP_CONVERTER_17,
    19: OP_CONVERTER_19,
}
