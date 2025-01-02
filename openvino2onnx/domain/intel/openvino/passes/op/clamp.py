"""
Copyright Wenyi Tang 2024-2025

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

import numpy as np
from onnx.helper import make_node
from onnx.mapping import TENSOR_TYPE_MAP
from onnx.onnx_pb import NodeProto

from openvino2onnx.graph import OnnxGraph
from openvino2onnx.passes.utils import make_constant

from . import OP_CONVERT, BaseNodeConversion


@OP_CONVERT.register(name="clamp")
class Clamp(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/activation/clamp-1.html

    https://onnx.ai/onnx/operators/onnx__Clip.html
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        min_value = float(self.get_attribute(ori_node, "min"))  # type: ignore
        max_value = float(self.get_attribute(ori_node, "max"))  # type: ignore
        try:
            prec = graph.tensor_type(ori_node.output[0])
        except ValueError:
            prec = graph.tensor_type(ori_node.input[0])
        dtype = TENSOR_TYPE_MAP[prec].np_dtype
        min_value = make_constant(
            f"{ori_node.name}/min", np.array(min_value, dtype=dtype)
        )
        max_value = make_constant(
            f"{ori_node.name}/max", np.array(max_value, dtype=dtype)
        )
        ori_node.input.extend([min_value.output[0], max_value.output[0]])
        self += [min_value, max_value]
        return make_node(
            "Clip", inputs=ori_node.input, outputs=ori_node.output, name=ori_node.name
        )
