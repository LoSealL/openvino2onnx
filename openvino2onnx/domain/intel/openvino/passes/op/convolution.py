"""
Copyright Wenyi Tang 2024-2025

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

import itertools
from typing import Any, Dict

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
        dilations = self.get_attribute(ori_node, "dilations")
        assert isinstance(dilations, str) or dilations is None
        dilations = text_to_integers(dilations)
        strides = self.get_attribute(ori_node, "strides")
        assert isinstance(strides, str) or strides is None
        strides = text_to_integers(strides)
        pads_begin = self.get_attribute(ori_node, "pads_begin")
        assert isinstance(pads_begin, str) or pads_begin is None
        pads_begin = text_to_integers(pads_begin)
        pads_end = self.get_attribute(ori_node, "pads_end")
        assert isinstance(pads_end, str) or pads_end is None
        pads_end = text_to_integers(pads_end)
        auto_pad = _trans_auto_pad(self.get_attribute(ori_node, "auto_pad"))
        attrs: Dict[str, Any] = dict(dilations=dilations, strides=strides)
        if auto_pad == "NOTSET":
            assert pads_begin is not None and pads_end is not None
            attrs["pads"] = list(itertools.chain(pads_begin, pads_end))
        else:
            attrs["auto_pad"] = auto_pad
        kernel_shape = graph.tensor_shape(ori_node.input[1])
        k0, k1 = kernel_shape[-2:]
        if isinstance(k0, int) and isinstance(k1, int):
            attrs["kernel_shape"] = [k0, k1]
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
        dilations = self.get_attribute(ori_node, "dilations")
        assert isinstance(dilations, str) or dilations is None
        dilations = text_to_integers(dilations)
        strides = self.get_attribute(ori_node, "strides")
        assert isinstance(strides, str) or strides is None
        strides = text_to_integers(strides)
        pads_begin = self.get_attribute(ori_node, "pads_begin")
        assert isinstance(pads_begin, str) or pads_begin is None
        pads_begin = text_to_integers(pads_begin)
        pads_end = self.get_attribute(ori_node, "pads_end")
        assert isinstance(pads_end, str) or pads_end is None
        pads_end = text_to_integers(pads_end)
        auto_pad = _trans_auto_pad(self.get_attribute(ori_node, "auto_pad"))
        attrs: Dict[str, Any] = dict(dilations=dilations, strides=strides)
        if auto_pad == "NOTSET":
            assert pads_begin is not None and pads_end is not None
            attrs["pads"] = list(itertools.chain(pads_begin, pads_end))
        else:
            attrs["auto_pad"] = auto_pad
        # calculate groups
        weight_shape = graph.tensor_shape(ori_node.input[1])
        group = weight_shape[0]
        if isinstance(group, str) or group < 1:
            raise ValueError(f"Weight shape[0] is dynamic: {group}")
        attrs["group"] = group
        if group > 1:
            # should reshape weights from [G, C_out, C_in, K0, K1, ...]
            # to [G * C_out, C_in, K0, K1, ...].
            weights = self.get_value(ori_node.input[1])
            assert weights is not None
            weights = weights.reshape([-1, *weights.shape[2:]])
            cst_node = make_constant(f"{ori_node.name}/weights", weights)
            ori_node.input[1] = cst_node.output[0]
            weight_shape = weights.shape
            self += cst_node
        k0, k1 = weight_shape[-2:]
        if isinstance(k0, int) and isinstance(k1, int):
            attrs["kernel_shape"] = [k0, k1]
        return make_node(
            "Conv",
            inputs=ori_node.input,
            outputs=ori_node.output,
            name=ori_node.name,
            **attrs,
        )
