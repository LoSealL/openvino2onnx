"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from typing import List

import numpy as np
from onnx.helper import tensor_dtype_to_np_dtype
from onnx.onnx_pb import NodeProto

from openvino2onnx.graph import OnnxGraph
from openvino2onnx.passes import PASSES, logger
from openvino2onnx.passes.pattern import SingleNodePattern
from openvino2onnx.passes.rewriter import Rewriter
from openvino2onnx.passes.utils import make_constant


@PASSES.register(
    "canonicalize_negative_zero_point", deps=["canonicalize_negative_scale"]
)
class CanonicalizeNegativeZeroPointRewriter(Rewriter):
    """Negative zero point is not supported on NPU-NCE engine.
    For negative zero point, we should increase the scale to enlarge the range
    of the data.

    Before:

        x [1, 5] (float32) / 0.01569 - 64 = qx [0, 255]
        (scale [0.01569], zero_point [-64])

    After:

        x [1, 5] (float32) / 0.01960 - 0 = qx [0, 255]
        (scale [0.01960], zero_point [0])
    """

    def __init__(self):
        super().__init__(pattern=SingleNodePattern("DequantizeLinear"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        node = nodes[0]
        scale = self.get_value(node.input[1])
        zero_point = self.get_value(node.input[2])
        if zero_point is None or np.all(zero_point >= 0):
            return
        if scale is None:
            raise ValueError(
                f"encounter node {node.name} with negative zero point but "
                "nonconst scale"
            )
        qtype = tensor_dtype_to_np_dtype(graph.tensor_type(node.input[2]))
        bits = np.iinfo(qtype).bits
        qmax = 2**bits - 1
        x_max = (qmax - zero_point) * scale
        x_min = np.minimum((0 - zero_point) * scale, 0)  # x_min should below zero
        new_scale = (x_max - x_min) / qmax
        new_zero_point = np.maximum(zero_point, 0)
        logger.warning(
            f"change {node.name}: scale {scale} -> {new_scale}, "
            f"zero_point {zero_point} -> {new_zero_point}"
        )

        scale_cst = make_constant(f"{node.name}/scale", new_scale)
        zp_cst = make_constant(f"{node.name}/zero_point", new_zero_point)

        node.input[1] = scale_cst.output[0]
        node.input[2] = zp_cst.output[0]

        if prev_n := self.get_input_node(node, 0):
            if prev_n.op_type in ("QuantizeLinear",):
                prev_n.input[1] = scale_cst.output[0]
                prev_n.input[2] = zp_cst.output[0]

        self += [scale_cst, zp_cst]
