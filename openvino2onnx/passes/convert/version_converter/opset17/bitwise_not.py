"""
Copyright Wenyi Tang 2024

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
from openvino2onnx.passes.utils import make_constant

from . import OP_CONVERTER


@OP_CONVERTER.register("BitwiseNot")
class BitwiseNot(Rewriter):
    """Rewrite bitwise as arithmetic operators"""

    def __init__(self):
        super().__init__(SingleNodePattern("BitwiseNot"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto], *args, **kwargs):
        node = nodes[0]
        dtype = graph.tensor_type(node.input[0])
        dtype = tensor_dtype_to_np_dtype(dtype)
        utype = dtype.str.replace("i", "u")
        cst_node = make_constant(
            f"{node.name}/not", np.array(np.iinfo(utype).max, utype)
        )
        if dtype != utype:
            cast_node = make_node(
                "Cast",
                inputs=[node.input[0]],
                outputs=[f"{node.name}/Cast_output0"],
                name=f"{node.name}/Cast",
                to=np_dtype_to_tensor_dtype(np.dtype(utype)),
            )
            node.input[0] = cast_node.output[0]
            self += cast_node
        op_node = make_node(
            "Sub",
            inputs=[cst_node.output[0], node.input[0]],
            outputs=node.output,
            name=node.name,
        )
        if dtype != utype:
            cast_back_node = make_node(
                "Cast",
                inputs=[f"{node.name}/Cast_back_input0"],
                outputs=op_node.output,
                name=f"{node.name}/Cast_back",
                to=np_dtype_to_tensor_dtype(dtype),
            )
            op_node.output[0] = cast_back_node.input[0]
            self += cast_back_node
        self -= node
        self += [cst_node, op_node]
