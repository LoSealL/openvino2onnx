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

from . import OP_CONVERT, BaseNodeConversion


@OP_CONVERT.register(name="avg_pool")
class AvgPool(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/pooling/avg-pool-14.html

    https://onnx.ai/onnx/operators/onnx__AveragePool.html
    """

    def _trans_ceil_mode(self, ir_mode):
        if ir_mode == "ceil":
            return 1
        return 0

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        ceil_mode = self._trans_ceil_mode(self.get_attribute(ori_node, "rounding_type"))
        pads_begin = text_to_integers(self.get_attribute(ori_node, "pads_begin"))
        pads_end = text_to_integers(self.get_attribute(ori_node, "pads_end"))
        dilations = text_to_integers(self.get_attribute(ori_node, "dilations"))
        strides = text_to_integers(self.get_attribute(ori_node, "strides"))
        kernel = text_to_integers(self.get_attribute(ori_node, "kernel"))
        pads = list(itertools.chain(pads_begin, pads_end))
        if auto_pad := self.get_attribute(ori_node, "auto_pad"):
            if auto_pad == "explicit":
                auto_pad = "NOTSET"
            else:
                pads = None
        return make_node(
            "AveragePool",
            inputs=ori_node.input,
            outputs=ori_node.output,
            name=ori_node.name,
            auto_pad=auto_pad.upper(),
            ceil_mode=ceil_mode,
            dilations=dilations,
            pads=pads,
            strides=strides,
            kernel_shape=kernel,
        )
