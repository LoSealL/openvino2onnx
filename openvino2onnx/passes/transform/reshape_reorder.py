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
from openvino2onnx.passes import PASSES
from openvino2onnx.passes.pattern import GraphPattern, SingleNodePattern
from openvino2onnx.passes.rewriter import Rewriter


@PASSES.register(name="reshape_reorder")
class ReshapeReorderRewrite(Rewriter):
    """reorder Reshape after “DequantizeLinear” and eliminate duplicated reshape."""

    def __init__(self):
        pattern = GraphPattern()
        reshape1 = SingleNodePattern("Reshape")
        quantize = SingleNodePattern("QuantizeLinear")
        dequantize = SingleNodePattern("DequantizeLinear")

        pattern.add_edge(reshape1, quantize)
        pattern.add_edge(quantize, dequantize)

        super().__init__(pattern)

    def check(self, graph, reshape1_node, reshape2_node):
        """check shapes between reshape1 input tensor and reshape2 output tensor"""
        if reshape2_node is None or reshape2_node.op_type != "Reshape":
            return False

        # it maybe None if it's generated from other pass
        shape1 = graph.tensor_shape(reshape1_node.input[0])
        shape2 = self.get_value(self.get_input_node(reshape2_node, 1))
        valid_shape = shape1 is not None and len(shape1) == len(shape2)
        return valid_shape and tuple(shape1) == tuple(shape2)

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        reshape1_node, quant_node, dequant_node = nodes
        # remove first reshape
        new_quant_node = make_node(
            op_type="QuantizeLinear",
            inputs=[reshape1_node.input[0]] + quant_node.input[1:],
            outputs=quant_node.output[:],
            name=quant_node.name + "_rr",
        )

        # eliminate duplicated reshape
        post_dq_nodes = graph.onnx_successors(dequant_node)
        if len(post_dq_nodes) != 1:
            return
        reshape2_node = post_dq_nodes[0]
        if self.check(graph, reshape1_node, reshape2_node):
            new_dequant_node = make_node(
                op_type="DequantizeLinear",
                inputs=dequant_node.input[:],
                outputs=reshape2_node.output[:],
                name=dequant_node.name + "_rr",
            )
            self += [new_quant_node, new_dequant_node]
            self -= [reshape1_node, quant_node, dequant_node, reshape2_node]
            # remove shape constant
            self -= [
                self.get_input_node(reshape1_node, 1),
                self.get_input_node(reshape2_node, 1),
            ]
        else:
            new_dequant_node = make_node(
                op_type="DequantizeLinear",
                inputs=dequant_node.input[:],
                outputs=[dequant_node.output[0] + "_output"],
                name=dequant_node.name + "_rr",
            )
            new_rehsape_node = make_node(
                op_type="Reshape",
                inputs=[dequant_node.output[0] + "_output", reshape1_node.input[1]],
                outputs=[dequant_node.output[0]],
                name=reshape1_node.name + "_rr",
            )
            self += [new_quant_node, new_dequant_node, new_rehsape_node]
            self -= [reshape1_node, quant_node, dequant_node]
