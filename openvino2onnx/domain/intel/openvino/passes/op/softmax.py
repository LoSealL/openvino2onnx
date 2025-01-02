"""
Copyright Wenyi Tang 2024-2025

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from openvino2onnx.graph import OnnxGraph

from . import OP_CONVERT, BaseNodeConversion


@OP_CONVERT.register(name="softmax")
class SoftMax(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/activation/softmax-8.html

    https://onnx.ai/onnx/operators/onnx__Softmax.html
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        axis = self.get_attribute(ori_node, "axis") or -1
        assert isinstance(axis, (str, int))
        axis = int(axis)
        return make_node(
            "Softmax",
            inputs=ori_node.input,
            outputs=ori_node.output,
            name=ori_node.name,
            axis=axis,
        )
