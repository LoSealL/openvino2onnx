"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from openvino2onnx.passes import PASSES, Registry

IR_PASSES = Registry("IR_PASS", parent=PASSES)
