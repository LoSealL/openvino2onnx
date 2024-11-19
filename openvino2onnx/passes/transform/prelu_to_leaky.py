"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

# pylint: disable=arguments-differ

from typing import List

from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from openvino2onnx.graph import OnnxGraph
from openvino2onnx.passes import L1
from openvino2onnx.passes.pattern import SingleNodePattern
from openvino2onnx.passes.rewriter import Rewriter


@L1.register(name="prelu_to_leaky", deps=["initializer_to_constant"])
class PReluToLeakyRewriter(Rewriter):
    """Convert PRelu whose parameter is a scaler to LeakyRelu."""

    def __init__(self):
        super().__init__(pattern=SingleNodePattern("PRelu"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto], *args, **kwargs):
        node = nodes[0]
        # quick check slope shape
        slope_shape = graph.tensor_shape(node.input[1])
        if len(slope_shape) >= 1 and slope_shape[0] != 1:
            # keep PRelu
            return

        slope_value = None
        for pred in graph.onnx_predecessors(node):
            if node.input[1] in pred.output:
                slope_value = self.get_value(pred)
                break
        if slope_value is None:
            # slope is not static
            return

        leakyrelu_node = make_node(
            "LeakyRelu",
            inputs=node.input[:1],
            outputs=node.output,
            name=node.name + "/to_leaky",
            alpha=float(slope_value.squeeze()),
        )
        self += leakyrelu_node
        self -= node
