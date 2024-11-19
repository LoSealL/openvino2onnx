"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

# check onnxscript and torch installation

try:
    import torch

    if torch.__version__ < "2.0.0":
        raise ImportError("PyTorch version should be 2.0.0 or higher.")
except ImportError as e:
    raise ImportError("PyTorch>=2.0.0 is not installed.") from e


try:
    import onnxscript
except ImportError as e:
    raise ImportError("onnxscript is not installed.") from e


from onnx import OperatorSetIdProto
from onnx.helper import make_operatorsetid

IR_DOMAIN: OperatorSetIdProto = make_operatorsetid("ai.pytorch.aten", 2)
"""Pytorch ATen"""
