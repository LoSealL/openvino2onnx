"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from openvino2onnx.graph import OnnxGraph

from . import OP_CONVERT, BaseNodeConversion


@OP_CONVERT.register(name="hsigmoid")
class HSigmoid(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/activation/hsigmoid-5.html

    https://onnx.ai/onnx/operators/onnx__HardSigmoid.html
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        return make_node(
            "HardSigmoid",
            inputs=ori_node.input,
            outputs=ori_node.output,
            name=ori_node.name,
            alpha=1 / 6,
            beta=0.5,
        )
