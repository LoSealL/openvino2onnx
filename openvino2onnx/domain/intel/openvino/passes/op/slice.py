"""
Copyright Wenyi Tang 2025

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from openvino2onnx.graph import OnnxGraph

from . import OP_CONVERT, BaseNodeConversion


@OP_CONVERT.register(name="slice")
class Slice(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/movement/slice-8.html

    https://onnx.ai/onnx/operators/onnx__Slice.html
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        inputs = ori_node.input
        if len(inputs) == 4:
            # data, starts, ends, steps
            inputs.append(inputs[-4])
            inputs[3] = ""
        elif len(inputs) == 5:
            # data, starts, ends, steps, axes
            inputs[3], inputs[4] = inputs[4], inputs[3]
        return make_node(
            "Slice",
            inputs=inputs,
            outputs=ori_node.output,
            name=ori_node.name,
        )
