"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from typing import List

from onnx.onnx_pb import NodeProto, TensorProto

from openvino2onnx.graph import OnnxGraph
from openvino2onnx.passes.pattern import SingleNodePattern
from openvino2onnx.passes.rewriter import Rewriter

from . import IR_PASSES


@IR_PASSES.register("eliminate_useless_floor")
class EliminateUselessFloorRewriter(Rewriter):
    """OpenVINO has a bug that floor on integers, it should be removed from the graph.

    Before:

        x(i64) -> Floor -> y(i64)

    After:

        x(i64) -> (Identity) -> y(i64)
    """

    def __init__(self):
        super().__init__(pattern=SingleNodePattern("Floor"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        node = nodes[0]
        _, input_type = graph.tensor_info(node.input[0])
        if input_type is None:
            return
        elif input_type in (
            TensorProto.FLOAT,
            TensorProto.FLOAT16,
            TensorProto.DOUBLE,
            TensorProto.BFLOAT16,
        ):
            return

        for next_n in self.get_output_node(node, 0):
            ind = 0
            for ind, input_name in enumerate(next_n.input):
                if input_name == node.output[0]:
                    break
            next_n.input[ind] = node.input[0]
        self -= node
