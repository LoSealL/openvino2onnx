"""
Copyright Wenyi Tang 2024-2025

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from typing import List

import numpy as np
from onnx.helper import make_node, np_dtype_to_tensor_dtype, tensor_dtype_to_np_dtype
from onnx.onnx_pb import NodeProto

from openvino2onnx.graph import OnnxGraph
from openvino2onnx.passes.pattern import SingleNodePattern
from openvino2onnx.passes.rewriter import Rewriter
from openvino2onnx.passes.utils import cast_in, cast_out, make_constant

from . import OP_CONVERTER


@OP_CONVERTER.register("BitwiseNot")
class BitwiseNot(Rewriter):
    """Rewrite bitwise as arithmetic operators"""

    def __init__(self):
        super().__init__(SingleNodePattern("BitwiseNot"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto], *args, **kwargs):
        node = nodes[0]
        etype = graph.tensor_type(node.input[0])
        dtype = tensor_dtype_to_np_dtype(etype)
        utype = dtype.str.replace("i", "u")
        cst_node = make_constant(
            f"{node.name}/not", np.array(np.iinfo(utype).max, utype)
        )
        if dtype != utype:
            self += cast_in(node, 0, np_dtype_to_tensor_dtype(np.dtype(utype)))
        op_node = make_node(
            "Sub",
            inputs=[cst_node.output[0], node.input[0]],
            outputs=node.output,
            name=node.name,
        )
        if dtype != utype:
            self += cast_out(op_node, 0, np_dtype_to_tensor_dtype(dtype))
        self -= node
        self += [cst_node, op_node]
