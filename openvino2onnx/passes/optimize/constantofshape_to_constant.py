"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from typing import List

import numpy as np
from onnx import NodeProto

from openvino2onnx import OnnxGraph
from openvino2onnx.passes import L1
from openvino2onnx.passes.pattern import SingleNodePattern
from openvino2onnx.passes.rewriter import Rewriter
from openvino2onnx.passes.utils import make_constant


@L1.register(name="constantofshape_to_constant")
class ConstantOfShapeRewriter(Rewriter):
    """Rewrite ConstantOfShape to a constant node."""

    def __init__(self):
        super().__init__(pattern=SingleNodePattern("ConstantOfShape"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        node = nodes[0]
        shape = self.get_value(node.input[0])
        value = self.get_attribute(node, "value")
        if value is None:  # default value is 0.0
            value = np.array([0], dtype=np.float32)
        assert len(value) == 1
        value = value[0]
        const_node = make_constant(
            node.name + "/const", np.zeros(shape, dtype=value.dtype) + value
        )
        const_node.output[0] = node.output[0]
        self -= node
        self += const_node
