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


def _trans_auto_pad(ir_str: str):
    match ir_str.lower():
        case "explicit":
            return "NOTSET"
        case "same_lower":
            return "SAME_LOWER"
        case "same_upper":
            return "SAME_UPPER"
        case "valid":
            return "VALID"
    raise ValueError(f"Unknown auto_pad={ir_str}")


@OP_CONVERT.register(name="convolution_backprop_data")
class ConvolutionBackpropData(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/convolution/convolution-backprop-data-1.html

    https://onnx.ai/onnx/operators/onnx__ConvTranspose.html
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        strides = text_to_integers(self.get_attribute(ori_node, "strides"))
        pads_begin = text_to_integers(self.get_attribute(ori_node, "pads_begin"))
        pads_end = text_to_integers(self.get_attribute(ori_node, "pads_end"))
        dilations = text_to_integers(self.get_attribute(ori_node, "dilations"))
        auto_pad = _trans_auto_pad(self.get_attribute(ori_node, "auto_pad"))
        output_padding = text_to_integers(
            self.get_attribute(ori_node, "output_padding")
        )
        attrs = dict(
            dilations=dilations,
            group=1,
            output_padding=output_padding,
            strides=strides,
        )
        if auto_pad == "NOTSET":
            attrs["pads"] = itertools.chain(pads_begin, pads_end)
        else:
            attrs["auto_pad"] = auto_pad
        if len(ori_node.input) > 2:
            output_shape = self.get_value(ori_node.input[2])
            attrs["output_shape"] = output_shape.tolist()
            ori_node.input.pop(2)

        return make_node(
            "ConvTranspose",
            inputs=ori_node.input,
            outputs=ori_node.output,
            name=ori_node.name,
            **attrs,
        )


@OP_CONVERT.register(name="group_convolution_backprop_data")
class GroupConvolutionBackpropData(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/convolution/group-convolution-backprop-data-1.html

    https://onnx.ai/onnx/operators/onnx__ConvTranspose.html
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        strides = text_to_integers(self.get_attribute(ori_node, "strides"))
        pads_begin = text_to_integers(self.get_attribute(ori_node, "pads_begin"))
        pads_end = text_to_integers(self.get_attribute(ori_node, "pads_end"))
        dilations = text_to_integers(self.get_attribute(ori_node, "dilations"))
        auto_pad = _trans_auto_pad(self.get_attribute(ori_node, "auto_pad"))
        output_padding = text_to_integers(
            self.get_attribute(ori_node, "output_padding")
        )
        attrs = dict(
            dilations=dilations,
            output_padding=output_padding,
            strides=strides,
        )
        if auto_pad == "NOTSET":
            attrs["pads"] = itertools.chain(pads_begin, pads_end)
        else:
            attrs["auto_pad"] = auto_pad
        if len(ori_node.input) > 2:
            output_shape = self.get_value(ori_node.input[2])
            attrs["output_shape"] = output_shape.tolist()
            ori_node.input.pop(2)
        # calculate groups
        weight_shape = graph.tensor_shape(ori_node.input[1])
        group = weight_shape[0]
        attrs["group"] = group
        if group > 1:
            # should reshape weights from [G, C_in, C_out, K0, K1, ...]
            # to [G * C_out, C_in, K0, K1, ...].
            weights = self.get_value(ori_node.input[1])
            weights = weights.reshape([-1, *weights.shape[2:]])
            cst_node = make_constant(f"{ori_node.name}/weights", weights)
            ori_node.input[1] = cst_node.output[0]
            self += cst_node

        return make_node(
            "ConvTranspose",
            inputs=ori_node.input,
            outputs=ori_node.output,
            name=ori_node.name,
            **attrs,
        )
