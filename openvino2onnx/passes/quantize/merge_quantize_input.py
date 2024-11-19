"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

# pylint: disable=arguments-differ
from typing import List

from onnx.helper import make_tensor_value_info
from onnx.onnx_pb import NodeProto

from openvino2onnx.graph import OnnxGraph
from openvino2onnx.passes import PASSES
from openvino2onnx.passes.pattern import GraphPattern, SingleNodePattern
from openvino2onnx.passes.rewriter import Rewriter


@PASSES.register(name="merge_quantize_input")
class MergeQuantizeInputPass(Rewriter):
    """Merge QuantizeLinear into inputs and change input type to uint8.

    Example::

        Before:

            input{float32} - QuantizeLinear - DequantizeLinear

        After:

            input{uint8} - DequantizeLinear
    """

    def __init__(self):
        pattern = GraphPattern()
        qpattern = SingleNodePattern("QuantizeLinear")
        dqpattern = SingleNodePattern("DequantizeLinear")
        pattern.add_edge(qpattern, dqpattern)
        super().__init__(pattern=pattern)

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        qlinear, dqlinear = nodes
        if qlinear.input[0] not in graph.inputs:
            # only process the node after graph inputs
            return
        succ_nodes = graph.onnx_successors(qlinear)
        pred_nodes = graph.onnx_predecessors(qlinear)
        if len(succ_nodes) != 1 or succ_nodes[0].op_type != "DequantizeLinear":
            # double check
            return
        # quantized data type
        shape, qtype = graph.tensor_info(qlinear.output[0])
        # remove the original input
        old_inp = graph.input.pop(graph.inputs.pop(qlinear.input[0]))
        input_name = old_inp.name
        # keep input name unchanged
        dqlinear.input[0] = input_name
        graph.input.append(make_tensor_value_info(dqlinear.input[0], qtype, shape))
        graph.inputs[dqlinear.input[0]] = len(graph.input) - 1
        self -= qlinear
        self -= pred_nodes
