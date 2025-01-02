"""
Copyright Wenyi Tang 2024-2025

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from openvino2onnx.graph import OnnxGraph

from . import OP_CONVERT, BaseNodeConversion


@OP_CONVERT.register(name="gather_nd")
class GatherND(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/movement/gather-nd-8.html

    https://onnx.ai/onnx/operators/onnx__GatherND.html
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        batch_dims = self.get_attribute(ori_node, "batch_dims")
        assert isinstance(batch_dims, (int, float, str))
        return make_node(
            "GatherND",
            inputs=ori_node.input,
            outputs=ori_node.output,
            name=ori_node.name,
            batch_dims=int(batch_dims),
        )
