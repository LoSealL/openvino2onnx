"""
Copyright Wenyi Tang 2024-2025

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

import numpy as np
from onnx.helper import make_node
from onnx.onnx_pb import NodeProto, TensorProto

from openvino2onnx.graph import OnnxGraph
from openvino2onnx.passes.utils import cast_in, cast_out, make_constant

from . import OP_CONVERT, BaseNodeConversion


@OP_CONVERT.register(name="topk")
class TopK(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/sort/top-k-11.html

    https://onnx.ai/onnx/operators/onnx__TopK.html
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        k_type = graph.tensor_type(ori_node.input[1])
        if k_type == TensorProto.INT32:
            self += cast_in(ori_node, 1, TensorProto.INT64)
        k_shape = make_constant(f"{ori_node}/Kshape", np.array([1], dtype=np.int64))
        reshape_k = make_node(
            "Reshape",
            inputs=[ori_node.input[1], k_shape.output[0]],
            outputs=[f"{ori_node.name}/Reshape/K"],
            name=f"{ori_node.name}/Reshape",
        )
        ori_node.input[1] = reshape_k.output[0]
        self += [k_shape, reshape_k]
        index_element_type = self.get_attribute(ori_node, "index_element_type")
        if index_element_type == "i32":
            self += cast_out(ori_node, 1, TensorProto.INT32)
        axis = self.get_attribute(ori_node, "axis")
        assert isinstance(axis, (str, float, int))
        mode = self.get_attribute(ori_node, "mode")
        sort = self.get_attribute(ori_node, "sort")
        if sort == "index":
            raise ValueError("sort='index' is not supported in onnx.")
        return make_node(
            "TopK",
            inputs=ori_node.input,
            outputs=ori_node.output,
            name=ori_node.name,
            axis=int(axis),
            largest=1 if mode == "max" else 0,
            sorted=0 if sort == "none" else 0,
        )
