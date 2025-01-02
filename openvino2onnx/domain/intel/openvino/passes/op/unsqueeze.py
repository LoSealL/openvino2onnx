"""
Copyright Wenyi Tang 2024-2025

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

import numpy as np
import onnx
from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from openvino2onnx.graph import OnnxGraph

from . import OP_CONVERT, BaseNodeConversion


@OP_CONVERT.register(name="unsqueeze")
class Unsqueeze(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/shape/unsqueeze-1.html

    https://onnx.ai/onnx/operators/onnx__Unsqueeze.html
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        axes_type = graph.tensor_type(ori_node.input[1])
        axes_shape = graph.tensor_shape(ori_node.input[1])
        numerical_shape = list(filter(lambda x: isinstance(x, int), axes_shape))
        if np.prod(numerical_shape) == 0:  # type: ignore
            ori_node.input.pop(1)
        elif axes_type != onnx.TensorProto.INT64:
            # add a cast
            cast = make_node(
                "Cast",
                inputs=[ori_node.input[1]],
                outputs=[ori_node.input[1] + "_int64"],
                name=f"{ori_node.name}/Cast",
                to=onnx.TensorProto.INT64,
            )
            ori_node.input[1] = ori_node.input[1] + "_int64"
            self += cast
        return make_node(
            ori_node.op_type,
            inputs=ori_node.input,
            outputs=ori_node.output,
            name=ori_node.name,
        )
