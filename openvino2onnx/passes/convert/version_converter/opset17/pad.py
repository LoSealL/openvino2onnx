"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from typing import List

import numpy as np
from onnx.onnx_pb import NodeProto

from openvino2onnx.graph import OnnxGraph
from openvino2onnx.passes.pattern import SingleNodePattern
from openvino2onnx.passes.rewriter import Rewriter
from openvino2onnx.passes.utils import make_constant

from . import OP_CONVERTER


@OP_CONVERTER.register("Pad")
class Pad(Rewriter):
    """Remove axes input."""

    def __init__(self):
        super().__init__(SingleNodePattern("Pad"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto], *args, **kwargs):
        node = nodes[0]
        mode = self.get_attribute(node, "mode")
        if mode == "wrap":
            raise ValueError(f"mode {mode} is not supported below opset 18")
        if len(node.input) != 4:
            return
        axes = self.get_value(node.input[3])
        if axes is None:
            raise ValueError(f"axes of {node.name} is not constant")
        pads = self.get_value(node.input[1])
        if pads is None:
            raise ValueError(f"pads of {node.name} is not constant")
        pads = pads.reshape([2, -1])
        input_rank = len(graph.tensor_shape(node.input[0]))
        pads_expand = np.zeros([2, input_rank], np.int64)
        k = 0
        for i in range(input_rank):
            if i in axes:
                pads_expand[:, i] = pads[:, k]
                k += 1
        node.input.pop(3)
        pads_cst = make_constant(f"{node.name}/pads", pads_expand.flatten())
        node.input[1] = pads_cst.output[0]
        self += pads_cst
