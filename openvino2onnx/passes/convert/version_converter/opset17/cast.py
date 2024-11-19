"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from typing import List

from onnx.onnx_pb import NodeProto

from openvino2onnx.graph import OnnxGraph
from openvino2onnx.passes.pattern import SingleNodePattern
from openvino2onnx.passes.rewriter import Rewriter

from . import OP_CONVERTER


@OP_CONVERTER.register("Cast")
class Cast(Rewriter):
    """Remove "saturate" attribute from Cast op."""

    def __init__(self):
        super().__init__(SingleNodePattern("Cast"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto], *args, **kwargs):
        node = nodes[0]
        self.remove_attribute(node, "saturate")


@OP_CONVERTER.register("CastLike")
class CastLike(Rewriter):
    """Remove "saturate" attribute from CastLike op."""

    def __init__(self):
        super().__init__(SingleNodePattern("CastLike"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto], *args, **kwargs):
        node = nodes[0]
        self.remove_attribute(node, "saturate")
