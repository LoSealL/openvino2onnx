"""
Copyright Wenyi Tang 2024-2025

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

import numpy as np
from onnx.helper import make_node, np_dtype_to_tensor_dtype
from onnx.onnx_pb import NodeProto, TensorProto

from openvino2onnx.domain.intel.openvino.ir.mapping import ETYPE2DTYPE
from openvino2onnx.graph import OnnxGraph
from openvino2onnx.passes.utils import cast_in

from . import OP_CONVERT, BaseNodeConversion


@OP_CONVERT.register(name="range")
class Range(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/generation/range-4.html

    https://onnx.ai/onnx/operators/onnx__Range.html
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        output_type = self.get_attribute(ori_node, "output_type")
        if output_type is not None:
            assert output_type in ETYPE2DTYPE, f"Unsupported output_type: {output_type}"
            output_type = ETYPE2DTYPE[output_type]
            output_dtype = np_dtype_to_tensor_dtype(np.dtype(output_type))
        else:
            output_dtype = graph.tensor_type(ori_node.input[0])
        if output_dtype == TensorProto.UNDEFINED:
            output_dtype = TensorProto.INT64
        dtype0 = graph.tensor_type(ori_node.input[0])
        if dtype0 != output_dtype:
            self += cast_in(ori_node, 0, output_dtype)
        dtype1 = graph.tensor_type(ori_node.input[1])
        if dtype1 != output_dtype:
            self += cast_in(ori_node, 1, output_dtype)
        dtype2 = graph.tensor_type(ori_node.input[2])
        if dtype2 != output_dtype:
            self += cast_in(ori_node, 2, output_dtype)
        return make_node(
            "Range",
            inputs=ori_node.input,
            outputs=ori_node.output,
            name=ori_node.name,
        )
