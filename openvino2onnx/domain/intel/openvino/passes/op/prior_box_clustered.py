"""
Copyright Wenyi Tang 2024-2025

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from onnx import numpy_helper
from onnx.helper import make_node
from onnx.onnx_pb import NodeProto
from openvino.runtime import Model, compile_model
from openvino.runtime.opset8 import constant, prior_box_clustered

from openvino2onnx.domain.intel.openvino.utils import text_to_boolean
from openvino2onnx.graph import OnnxGraph

from . import OP_CONVERT, BaseNodeConversion


def _to_floats(text):
    if not text:
        return []
    return list(map(float, text.split(",")))


@OP_CONVERT.register(name="prior_box_clustered")
class PriorBoxClustered(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/detection/prior-box-clustered-1.html

    Fold to constant
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        attrs = dict(
            width=_to_floats(self.get_attribute(ori_node, "width")) or [1.0],
            height=_to_floats(self.get_attribute(ori_node, "height")) or [1.0],
            clip=text_to_boolean(self.get_attribute(ori_node, "clip")),  # type: ignore
            step=float(self.get_attribute(ori_node, "step") or 0),  # type: ignore
            step_w=float(self.get_attribute(ori_node, "step_w") or 0),  # type: ignore
            step_h=float(self.get_attribute(ori_node, "step_h") or 0),  # type: ignore
            offset=float(self.get_attribute(ori_node, "offset")),  # type: ignore
            variance=_to_floats(self.get_attribute(ori_node, "variance")),
        )
        output_size = self.get_value(ori_node.input[0])
        image_size = self.get_value(ori_node.input[1])
        output_size = constant(output_size, name="output_size")
        image_size = constant(image_size, name="image_size")
        pbc = prior_box_clustered(output_size, image_size, attrs)
        model = Model([pbc], [])
        compiled_model = compile_model(model, "CPU")
        value = compiled_model()[-1]

        return make_node(
            "Constant",
            [],
            outputs=ori_node.output,
            name=ori_node.name,
            value=numpy_helper.from_array(value),
        )
