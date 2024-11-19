"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

import numpy as np
from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from openvino2onnx.graph import OnnxGraph
from openvino2onnx.passes.utils import make_constant

from . import OP_CONVERT, BaseNodeConversion


@OP_CONVERT.register(name="pad")
class Pad(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/movement/pad-12.html

    https://onnx.ai/onnx/operators/onnx__Pad.html
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        mode = self.get_attribute(ori_node, "pad_mode") or "constant"
        if mode in ("symmetric",):
            raise ValueError(f"Pad {ori_node.name} has unsupported pad mode: {mode}")

        begin = self.get_value(ori_node.input[1])
        end = self.get_value(ori_node.input[2])
        if len(ori_node.input) > 3:
            const_value = self.get_value(ori_node.input[3])
            if const_value is not None:
                ori_node.input[2] = ori_node.input[3]
            ori_node.input.pop(3)
        pads = make_constant(f"{ori_node.name}/pads", np.concatenate([begin, end]))
        ori_node.input[1] = pads.output[0]
        ori_node.input.pop(2)
        self += pads
        return make_node(
            "Pad",
            inputs=ori_node.input,
            outputs=ori_node.output,
            name=ori_node.name,
            mode=mode,
        )
