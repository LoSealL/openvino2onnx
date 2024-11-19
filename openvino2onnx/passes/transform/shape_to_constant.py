"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

# pylint: disable=arguments-differ

from typing import List

import numpy as np
from onnx.onnx_pb import NodeProto

from openvino2onnx.graph import OnnxGraph
from openvino2onnx.passes import PASSES
from openvino2onnx.passes.pattern import SingleNodePattern
from openvino2onnx.passes.rewriter import Rewriter
from openvino2onnx.passes.utils import make_constant


@PASSES.register(name="shape_to_constant")
class ShapeToConstantPass(Rewriter):
    """Convert static Shape op output to Constant."""

    def __init__(self):
        super().__init__(pattern=SingleNodePattern("Shape"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        node = nodes[0]
        try:
            shape = graph.tensor_shape(node.input[0])
            if not shape or not all(isinstance(i, int) for i in shape):
                return  # dynamic shape
        except ValueError:
            # shape is not constant
            return
        # replace Shape with Constant
        shape_const = make_constant(node.name + "/Reshape", np.array(shape, "int64"))
        shape_const.output[0] = node.output[0]
        self -= node
        self += shape_const
