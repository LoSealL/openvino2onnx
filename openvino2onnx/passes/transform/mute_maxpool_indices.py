"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from typing import List

from onnx.onnx_pb import NodeProto

from openvino2onnx.graph import OnnxGraph
from openvino2onnx.passes import PASSES
from openvino2onnx.passes.pattern import SingleNodePattern
from openvino2onnx.passes.rewriter import Rewriter


@PASSES.register("mute_maxpool_indices")
class MuteMaxPoolIndicesRewriter(Rewriter):
    """Disable the 2nd output of MaxPool, a.k.a indices. If there is not any
    downstream node uses the indices.

    Some devices like NPU may not recognize the indices output of MaxPool.
    """

    def __init__(self):
        super().__init__(SingleNodePattern("MaxPool"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        node = nodes[0]
        if len(node.output) == 1:
            return

        if not self.get_output_node(node, 1):
            # no downstream node uses the indices
            node.output.pop(1)
