"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from openvino2onnx.graph import OnnxGraph

from . import OP_CONVERT, BaseNodeConversion


@OP_CONVERT.register(name="swish")
class Swish(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/activation/swish-4.html

    Swish(x) = x * sigmoid(x * beta)
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        sigmoid = make_node(
            "Sigmoid",
            inputs=[ori_node.input[0]],
            outputs=[f"{ori_node.name}/Sigmoid_output0"],
            name=f"{ori_node.name}/Sigmoid",
        )
        self += sigmoid
        if len(ori_node.output) > 1:
            # read beta
            beta_mul = make_node(
                "Mul",
                inputs=[ori_node.input[0], ori_node.input[1]],
                outputs=[f"{ori_node.name}/beta"],
                name=f"{ori_node.name}/beta",
            )
            sigmoid.input[0] = beta_mul.output[0]
            self += beta_mul

        return make_node(
            "Mul",
            inputs=[ori_node.input[0], sigmoid.output[0]],
            outputs=ori_node.output,
            name=ori_node.name,
        )
