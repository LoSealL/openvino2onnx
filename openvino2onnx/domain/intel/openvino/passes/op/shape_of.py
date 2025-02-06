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


@OP_CONVERT.register(name="shape_of")
class ShapeOf(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/shape/shape-of-3.html

    https://onnx.ai/onnx/operators/onnx__Shape.html
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        if out_dtype := self.get_attribute(ori_node, "output_type"):
            if out_dtype == "i32":
                # add a cast
                self += cast_in(ori_node, 0, TensorProto.INT32)
        return make_node(
            "Shape", inputs=ori_node.input, outputs=ori_node.output, name=ori_node.name
        )
