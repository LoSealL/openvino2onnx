"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from onnx.helper import make_node
from onnx.numpy_helper import from_array
from onnx.onnx_pb import NodeProto

from openvino2onnx.domain.intel.openvino.utils import text_to_integers
from openvino2onnx.graph import OnnxGraph

from . import OP_CONVERT, BaseNodeConversion


@OP_CONVERT.register(name="constant")
class Const(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/infrastructure/constant-1.html

    https://onnx.ai/onnx/operators/onnx__Constant.html
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        value = self.get_attribute(ori_node, "value")
        shape = self.get_attribute(ori_node, "shape")
        if shape is not None:
            shape = [i for i in text_to_integers(shape) if i != 0]
            value = value.reshape(shape)
        return make_node(
            "Constant",
            inputs=ori_node.input,
            outputs=ori_node.output,
            name=ori_node.name,
            value=from_array(value),
        )
