"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from onnx import OperatorSetIdProto
from onnx.helper import make_operatorsetid

IR_DOMAIN: OperatorSetIdProto = make_operatorsetid("ai.intel.openvino", 14)
"""OpenVINO IR v11 opset14"""
