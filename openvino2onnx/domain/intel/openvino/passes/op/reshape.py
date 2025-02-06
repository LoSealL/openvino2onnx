"""
Copyright Wenyi Tang 2024-2025

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from onnx.helper import make_node
from onnx.onnx_pb import NodeProto, TensorProto

from openvino2onnx.graph import OnnxGraph
from openvino2onnx.passes.utils import cast_in

from . import OP_CONVERT, BaseNodeConversion


@OP_CONVERT.register(name="reshape")
class Reshape(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/shape/reshape-1.html

    https://onnx.ai/onnx/operators/onnx__Reshape.html
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        shape_type = graph.tensor_type(ori_node.input[1])
        if shape_type != TensorProto.INT64:
            self += cast_in(ori_node, 1, TensorProto.INT64)
        return make_node(
            "Reshape",
            inputs=ori_node.input,
            outputs=ori_node.output,
            name=ori_node.name,
        )
