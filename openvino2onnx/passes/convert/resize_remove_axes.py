"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from typing import List

import numpy as np
from onnx.onnx_pb import NodeProto

from openvino2onnx.graph import OnnxGraph
from openvino2onnx.passes import PASSES
from openvino2onnx.passes.pattern import SingleNodePattern
from openvino2onnx.passes.rewriter import Rewriter
from openvino2onnx.passes.utils import make_constant


@PASSES.register("resize_remove_axes", deps=["resize_move_size_to_scale"])
class ResizeRemoveAxesRewriter(Rewriter):
    """Remove axes attribute in Resize operator.

    Since ``axes`` attribute is introduced since opset 18, many backends do not
    support it.

    @Deprecated
    Use ``downgrade_op_version`` instead.
    """

    def __init__(self):
        super().__init__(SingleNodePattern("Resize"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto], *args, **kwargs):
        node = nodes[0]
        if axes := self.get_attribute(node, "axes"):
            input_shape = graph.tensor_shape(node.input[0])
            input_rank = len(input_shape)
            axes = [i if i >= 0 else i + input_rank for i in axes]
            missing_axes = set(range(input_rank)) - set(axes)
            scales = self.get_input_node(node, 2)
            scales_value = self.get_value(scales)
            dtype = scales_value.dtype
            scales_value = {axis: scales_value[i] for i, axis in enumerate(axes)}
            scales_value.update({axis: 1.0 for axis in missing_axes})
            scales_value = [scales_value[axis] for axis in sorted(scales_value)]
            scales_value = np.array(scales_value, dtype=dtype)
            new_scales = make_constant(scales.name + "/new", scales_value)
            self.set_attribute(node, "axes", list(range(input_rank)))
            node.input[2] = new_scales.output[0]
            self -= scales
            self += new_scales
