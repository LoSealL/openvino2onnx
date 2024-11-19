"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from openvino2onnx.graph import OnnxGraph

from . import OP_CONVERT, BaseNodeConversion


@OP_CONVERT.register(name="hard_sigmoid")
class HardSigmoid(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/activation/hard-sigmoid-1.html

    https://onnx.ai/onnx/operators/onnx__HardSigmoid.html
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        alpha = self.get_value(ori_node.input[1])
        beta = self.get_value(ori_node.input[2])
        if alpha is None or beta is None:
            raise ValueError("alpha and beta must be constant values")
        return make_node(
            "HardSigmoid",
            inputs=ori_node.input,
            outputs=ori_node.output,
            name=ori_node.name,
            alpha=float(alpha),
            beta=float(beta),
        )
