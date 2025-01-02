"""
Copyright Wenyi Tang 2024-2025

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from typing import List

from onnx.onnx_pb import NodeProto

from openvino2onnx.graph import OnnxGraph
from openvino2onnx.passes.pattern import SingleNodePattern
from openvino2onnx.passes.rewriter import Rewriter

from . import OP_CONVERTER


@OP_CONVERTER.register("AveragePool")
class AveragePool(Rewriter):
    """Remove "dilations" attribute."""

    def __init__(self):
        super().__init__(SingleNodePattern("AveragePool"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto], *args, **kwargs):
        node = nodes[0]
        dilations = self.get_attribute(node, "dilations")
        if dilations is not None and any(d != 1 for d in dilations):  # type: ignore
            raise ValueError("Dilations not supported in AveragePool below opset 19.")
        self.remove_attribute(node, "dilations")
