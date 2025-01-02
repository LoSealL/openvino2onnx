"""
Copyright Wenyi Tang 2024-2025

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from onnx import numpy_helper
from onnx.helper import make_node
from onnx.mapping import TENSOR_TYPE_MAP
from onnx.onnx_pb import NodeProto
from openvino.runtime import Model, compile_model
from openvino.runtime.opset8 import constant, prior_box

from openvino2onnx.domain.intel.openvino.utils import text_to_boolean
from openvino2onnx.graph import OnnxGraph

from . import OP_CONVERT, BaseNodeConversion


def _to_floats(text):
    if not text:
        return []
    return list(map(float, text.split(",")))


@OP_CONVERT.register(name="prior_box")
class PriorBox(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/detection/prior-box-8.html

    Fold to constant
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        layer_shape = self.get_value(ori_node.input[0])
        image_shape = self.get_value(ori_node.input[1])
        if layer_shape is None or image_shape is None:
            return ori_node
        out_shape, out_dtype = graph.tensor_info(ori_node.output[0])
        layer_shape = constant(layer_shape, name="layer_shape")
        image_shape = constant(image_shape, name="image_shape")

        pb_attrs = dict(
            min_size=_to_floats(self.get_attribute(ori_node, "min_size")),
            max_size=_to_floats(self.get_attribute(ori_node, "max_size")),
            aspect_ratio=_to_floats(self.get_attribute(ori_node, "aspect_ratio")),
            density=_to_floats(self.get_attribute(ori_node, "density")),
            fixed_ratio=_to_floats(self.get_attribute(ori_node, "fixed_ratio")),
            fixed_size=_to_floats(self.get_attribute(ori_node, "fixed_size")),
            clip=text_to_boolean(self.get_attribute(ori_node, "clip")),  # type: ignore
            flip=text_to_boolean(self.get_attribute(ori_node, "flip")),  # type: ignore
            step=float(self.get_attribute(ori_node, "step")),  # type: ignore
            offset=float(self.get_attribute(ori_node, "offset") or 0),  # type: ignore
            variance=_to_floats(self.get_attribute(ori_node, "variance")),
            scale_all_sizes=text_to_boolean(
                self.get_attribute(ori_node, "scale_all_sizes")  # type: ignore
            ),
        )
        if order := self.get_attribute(ori_node, "min_max_aspect_ratios_order"):
            assert isinstance(order, str)
            pb_attrs["min_max_aspect_ratios_order"] = text_to_boolean(order)
        prior_node = prior_box(layer_shape, image_shape, pb_attrs)
        model = Model([prior_node], [])
        compiled_model = compile_model(model, "CPU")
        value = compiled_model()[-1]
        assert out_shape is not None and value.shape == tuple(out_shape)
        assert value.dtype == TENSOR_TYPE_MAP[out_dtype].np_dtype

        return make_node(
            "Constant",
            [],
            outputs=ori_node.output,
            name=ori_node.name,
            value=numpy_helper.from_array(value),
        )
