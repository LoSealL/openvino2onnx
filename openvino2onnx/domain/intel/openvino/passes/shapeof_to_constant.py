"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from typing import List

import numpy as np
from onnx.onnx_pb import NodeProto

from openvino2onnx.graph import OnnxGraph
from openvino2onnx.passes.pattern import SingleNodePattern
from openvino2onnx.passes.rewriter import Rewriter
from openvino2onnx.passes.utils import make_constant

from . import IR_PASSES


@IR_PASSES.register("shapeof_to_constant")
class ShapeOfToConstantRewriter(Rewriter):
    """Calculate the result of ShapeOf node and replace it with a Constant node.

    If shape is dynamic, do nothing.
    """

    def __init__(self):
        super().__init__(pattern=SingleNodePattern("ShapeOf"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        node = nodes[0]
        input_shape, _ = graph.tensor_info(node.input[0])
        if input_shape is None:
            return
        elif any(i <= 0 for i in input_shape):
            return  # dynamic shape

        self -= node
        if self.get_attribute(node, "output_type") == "i32":
            dtype = np.int32
        else:
            dtype = np.int64
        cst_node = make_constant(node.name, np.array(input_shape, dtype))
        cst_node.output[0] = node.output[0]
        self += cst_node
