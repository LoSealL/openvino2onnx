"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from typing import List

from onnx.onnx_pb import NodeProto

from openvino2onnx.graph import OnnxGraph
from openvino2onnx.passes import PASSES, logger
from openvino2onnx.passes.pattern import SingleNodePattern
from openvino2onnx.passes.rewriter import Rewriter


@PASSES.register("eliminate_reshape", deps=["infer_shape"])
class EliminateInputOutputReshapeRewriter(Rewriter):
    """Eliminate useless reshape operations."""

    def __init__(self):
        super().__init__(SingleNodePattern("Reshape"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto], *args, **kwargs):
        node = nodes[0]
        self._rewrite_input_reshape(graph, node)
        self._rewrite_output_reshape(graph, node)

    def _rewrite_input_reshape(self, graph: OnnxGraph, node: NodeProto):
        """remove the reshape if it is the input of the graph

        Before:
            input(shape1) -> reshape(shape2) -> conv

        After:
            input(shape2) -> conv
        """
        if node.input[0] in graph.inputs:
            input_name = node.input[0]
            graph.remove_input(input_name)
            for n in graph.onnx_successors(node):
                graph.set_input(n, node.output[0])
                input_pos = [i for i in n.input].index(node.output[0])
                logger.debug(f"set input {input_pos} of {n.name} to {input_name}")
                graph.rename_input(node.output[0], input_name)
                n.input[input_pos] = input_name
            self -= node

    def _rewrite_output_reshape(self, graph: OnnxGraph, node: NodeProto):
        """remove the reshape if it is the output of the graph

        Before:
            conv(shape1) -> reshape(shape2) -> output(shape2)

        After:
            conv -> output(shape1)
        """
        if node.output[0] in graph.outputs:
            output_name = node.output[0]
            graph.remove_output(output_name)
            for n in graph.onnx_predecessors(node):
                graph.set_output(n, node.input[0])
                output_pos = [i for i in n.output].index(node.input[0])
                logger.debug(f"set output {output_pos} of {n.name} to {output_name}")
                graph.rename_output(node.input[0], output_name)
                n.output[output_pos] = output_name
            self -= node
