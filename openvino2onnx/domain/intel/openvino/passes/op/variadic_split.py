"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

import onnx
from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from openvino2onnx.graph import OnnxGraph

from . import OP_CONVERT, BaseNodeConversion


@OP_CONVERT.register(name="variadic_split")
class VariadicSplit(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/movement/variadic-split-1.html

    https://onnx.ai/onnx/operators/onnx__Split.html
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        axis = self.get_value(ori_node.input[1])
        ori_node.input.pop(1)
        dtype = graph.tensor_type(ori_node.input[-1])
        if dtype != onnx.TensorProto.INT64:
            cast = make_node(
                "Cast",
                inputs=[ori_node.input[-1]],
                outputs=[f"{ori_node.input[-1]}/Cast"],
                name=f"{ori_node.name}/Cast",
                to=onnx.TensorProto.INT64,
            )
            ori_node.input[-1] = cast.output[0]
            self += cast
        return make_node(
            "Split",
            inputs=ori_node.input,
            outputs=ori_node.output,
            name=ori_node.name,
            axis=int(axis),
        )
