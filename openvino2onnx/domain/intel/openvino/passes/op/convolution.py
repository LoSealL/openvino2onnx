"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

import itertools

from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from openvino2onnx.domain.intel.openvino.utils import text_to_integers
from openvino2onnx.graph import OnnxGraph
from openvino2onnx.passes.utils import make_constant

from . import OP_CONVERT, BaseNodeConversion


def _trans_auto_pad(ir_str):
    match ir_str:
        case "explicit" | "notset" | None:
            return "NOTSET"
        case "auto":
            return "SAME_LOWER"
        case "same_lower":
            return "SAME_LOWER"
        case "same_upper":
            return "SAME_UPPER"
        case "valid":
            return "VALID"
    raise ValueError(f"Unknown auto_pad={ir_str}")


@OP_CONVERT.register(name="convolution")
class Convolution(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/convolution/convolution-1.html

    https://onnx.ai/onnx/operators/onnx__Conv.html
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        dilations = text_to_integers(self.get_attribute(ori_node, "dilations"))
        strides = text_to_integers(self.get_attribute(ori_node, "strides"))
        pads_begin = text_to_integers(self.get_attribute(ori_node, "pads_begin"))
        pads_end = text_to_integers(self.get_attribute(ori_node, "pads_end"))
        auto_pad = _trans_auto_pad(self.get_attribute(ori_node, "auto_pad"))
        attrs = dict(dilations=dilations, strides=strides)
        if auto_pad == "NOTSET":
            attrs["pads"] = itertools.chain(pads_begin, pads_end)
        else:
            attrs["auto_pad"] = auto_pad
        return make_node(
            "Conv",
            inputs=ori_node.input,
            outputs=ori_node.output,
            name=ori_node.name,
            **attrs,
        )


@OP_CONVERT.register(name="group_convolution")
class GroupConvolution(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/convolution/group-convolution-1.html

    https://onnx.ai/onnx/operators/onnx__Conv.html
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        dilations = text_to_integers(self.get_attribute(ori_node, "dilations"))
        strides = text_to_integers(self.get_attribute(ori_node, "strides"))
        pads_begin = text_to_integers(self.get_attribute(ori_node, "pads_begin"))
        pads_end = text_to_integers(self.get_attribute(ori_node, "pads_end"))
        auto_pad = _trans_auto_pad(self.get_attribute(ori_node, "auto_pad"))
        attrs = dict(dilations=dilations, strides=strides)
        if auto_pad == "NOTSET":
            attrs["pads"] = itertools.chain(pads_begin, pads_end)
        else:
            attrs["auto_pad"] = auto_pad
        # calculate groups
        weight_shape = graph.tensor_shape(ori_node.input[1])
        group = weight_shape[0]
        attrs["group"] = group
        if group > 1:
            # should reshape weights from [G, C_out, C_in, K0, K1, ...]
            # to [G * C_out, C_in, K0, K1, ...].
            weights = self.get_value(ori_node.input[1])
            weights = weights.reshape([-1, *weights.shape[2:]])
            cst_node = make_constant(f"{ori_node.name}/weights", weights)
            ori_node.input[1] = cst_node.output[0]
            self += cst_node
        return make_node(
            "Conv",
            inputs=ori_node.input,
            outputs=ori_node.output,
            name=ori_node.name,
            **attrs,
        )
