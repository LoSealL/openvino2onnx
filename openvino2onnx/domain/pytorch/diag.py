"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

import numpy as np
import onnxscript
import torch
from onnx.numpy_helper import from_array
from onnxscript.onnx_opset import opset17 as op
from onnxscript.values import Opset
from torch.onnx._internal.jit_utils import GraphContext
from torch.onnx._internal.registration import onnx_symbolic

from . import IR_DOMAIN


@onnxscript.script(Opset(domain=IR_DOMAIN.domain, version=IR_DOMAIN.version))
def diag(x, diagonal: int):
    """Implementation of ``aten::diag`` using ONNX operators.

    Note:

        This implementation only supports 1D input
    """

    shape = op.Shape(x)
    idv = op.ConstantOfShape(shape, value=from_array(np.array([1], np.float32)))
    idv_x = op.CastLike(idv, x)
    idv1d_t = op.Unsqueeze(idv_x, axes=0)
    x_1d = op.Unsqueeze(x, axes=1)
    y = op.MatMul(x_1d, idv1d_t)
    mask = op.EyeLike(y, k=diagonal)
    y = op.Mul(y, mask)
    return y


@onnx_symbolic("aten::diag", 1, custom=True)
@torch.onnx.symbolic_helper.parse_args("v", "i")
def diag_symbolic(g: GraphContext, x: torch._C.Value, diagonal: int = 0):
    """Symbolic function for ``torch.diag``."""

    return g.onnxscript_op(diag, x, diagonal_i=diagonal).setTypeAs(x)
