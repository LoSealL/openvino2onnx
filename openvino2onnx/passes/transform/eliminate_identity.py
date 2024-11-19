"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

# pylint: disable=arguments-differ

from typing import List

from onnx.onnx_pb import NodeProto

from openvino2onnx.graph import OnnxGraph
from openvino2onnx.passes import L2
from openvino2onnx.passes.pattern import SingleNodePattern
from openvino2onnx.passes.rewriter import Rewriter
from openvino2onnx.passes.utils import make_constant


@L2.register(name="eliminate_identity")
class EliminateIdentityRewriter(Rewriter):
    """Eliminate Identity op.

    Before:

        constant -> identity -> fanout1
                           | -> fanout2
                           | -> ...

    After:

        constant_copy0 -> fanout1
        constant_copy1 -> fanout2
        constant_copy2 -> ...
    """

    def __init__(self):
        super().__init__(pattern=SingleNodePattern("Identity"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        identity_node = nodes[0]
        constant_node = self.get_input_node(identity_node, 0)
        if constant_node is None:
            # get value from initializer
            value = self.get_value(identity_node.input[0])
            fanout = 0
        else:
            value = self.get_value(constant_node)
            fanout = len(graph.onnx_successors(constant_node))
        if value is None:
            return  # not a constant

        for i, succ in enumerate(graph.onnx_successors(identity_node)):
            pos = list(succ.input).index(identity_node.output[0])
            # make a copy of constant
            constant_copy_node = make_constant(
                f"{identity_node.name}/constant_copy{i}",
                value,
            )
            succ.input[pos] = constant_copy_node.output[0]
            self += constant_copy_node

        self -= identity_node
        if fanout == 1:
            self -= constant_node
