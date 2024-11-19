"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from openvino2onnx.graph import OnnxGraph

from . import OP_CONVERT, BaseNodeConversion


@OP_CONVERT.register(name="split")
class Split(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/movement/split-1.html

    https://onnx.ai/onnx/operators/onnx__Split.html
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        axis = self.get_value(ori_node.input[1])
        num_splits = self.get_attribute(ori_node, "num_splits")
        return make_node(
            "Split",
            inputs=[ori_node.input[0]],
            outputs=ori_node.output,
            name=ori_node.name,
            axis=int(axis),
            num_outputs=int(num_splits),
        )
