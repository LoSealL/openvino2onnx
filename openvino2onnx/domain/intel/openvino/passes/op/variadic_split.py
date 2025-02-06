"""
Copyright Wenyi Tang 2024-2025

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

import onnx
from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from openvino2onnx.graph import OnnxGraph
from openvino2onnx.passes.utils import cast_in

from . import OP_CONVERT, BaseNodeConversion


@OP_CONVERT.register(name="variadic_split")
class VariadicSplit(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/movement/variadic-split-1.html

    https://onnx.ai/onnx/operators/onnx__Split.html
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        axis = self.get_value_or_die(ori_node.input[1])
        ori_node.input.pop(1)
        dtype = graph.tensor_type(ori_node.input[-1])
        if dtype != onnx.TensorProto.INT64:
            self += cast_in(ori_node, -1, onnx.TensorProto.INT64)
        return make_node(
            "Split",
            inputs=ori_node.input,
            outputs=ori_node.output,
            name=ori_node.name,
            axis=int(axis),
        )
