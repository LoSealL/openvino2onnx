"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from typing import List

from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from openvino2onnx.graph import OnnxGraph
from openvino2onnx.passes.pattern import SingleNodePattern
from openvino2onnx.passes.rewriter import Rewriter

from . import OP_CONVERTER


@OP_CONVERTER.register("Mish")
class Mish(Rewriter):
    """Convert Mish according to expression:

    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    """

    def __init__(self):
        super().__init__(SingleNodePattern("Mish"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        node = nodes[0]
        softplus = make_node(
            "Softplus",
            [node.input[0]],
            [f"{node.name}/Softplus_output0"],
            name=f"{node.name}/Softplus",
        )
        tanh_softplus = make_node(
            "Tanh",
            [softplus.output[0]],
            [f"{node.name}/Tanh_output0"],
            name=f"{node.name}/Tanh",
        )
        mul = make_node(
            "Mul",
            [node.input[0], tanh_softplus.output[0]],
            [node.output[0]],
            name=f"{node.name}/Mul",
        )
        self += [softplus, tanh_softplus, mul]
        self -= node
