"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from openvino2onnx.graph import OnnxGraph

from . import OP_CONVERT, BaseNodeConversion


@OP_CONVERT.register(name="transpose")
class Transpose(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/movement/transpose-1.html

    https://onnx.ai/onnx/operators/onnx__Transpose.html
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        perm = self.get_value(ori_node.input[1])
        if perm is None:
            raise RuntimeError(
                f"Do not support transpose ({ori_node.name}) with dynamic perm"
            )
        ori_node.input.pop(1)
        return make_node(
            "Transpose",
            inputs=ori_node.input,
            outputs=ori_node.output,
            name=ori_node.name,
            perm=list(map(int, perm)),
        )
