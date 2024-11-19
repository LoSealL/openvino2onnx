"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from abc import abstractmethod
from copy import deepcopy
from typing import List

from onnx.onnx_pb import NodeProto

from openvino2onnx.domain.intel.openvino.passes import IR_PASSES
from openvino2onnx.graph import OnnxGraph
from openvino2onnx.passes import Registry
from openvino2onnx.passes.pattern import SingleNodePattern
from openvino2onnx.passes.rewriter import Rewriter, RewriterRepeat


class BaseNodeConversion(Rewriter):
    """Abstract class for a single IR node conversion."""

    def __init__(self, pattern=None, repeat=RewriterRepeat.ONCE):
        if pattern is None:
            pattern = SingleNodePattern(self.__class__.__name__).with_attr("version")
        super().__init__(pattern=pattern, repeat=repeat)

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto], *args, **kwargs):
        node = nodes[0]
        new_node = self.replace(graph, deepcopy(node))
        self += new_node

    @abstractmethod
    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        """Replace the ori_node to ai.onnx operator"""
        raise NotImplementedError("replace method is not implemented")


OP_CONVERT = Registry("OP_CONVERT", parent=IR_PASSES)
