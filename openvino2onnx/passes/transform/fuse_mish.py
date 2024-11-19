"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

# pylint: disable=arguments-differ

from typing import List

from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from openvino2onnx.graph import OnnxGraph
from openvino2onnx.passes import PASSES
from openvino2onnx.passes.pattern import GraphPattern, SingleNodePattern
from openvino2onnx.passes.rewriter import Rewriter


@PASSES.register("fuse_mish")
class FuseMishRewriter(Rewriter):
    r"""Fuse Mish activation function.

     Mish is defined as:

    .. math::

         mish(x) = x \cdot tanh(softplus(x))

     where :math:`softplus(x) = ln(1 + exp(x))`.

     Before:

         a->softplus->tanh->mul->b
          \_________________/

     After:

         a->mish(function)->b
    """

    def __init__(self):
        pattern = GraphPattern()
        node_a = SingleNodePattern()
        softp = SingleNodePattern("Softplus")
        tanh = SingleNodePattern("Tanh")
        mul = SingleNodePattern("Mul")
        pattern.add_edge(node_a, softp)
        pattern.add_edge(softp, tanh)
        pattern.add_edge(tanh, mul)
        pattern.add_edge(node_a, mul)
        super().__init__(pattern)

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        _, softplus, tanh, mul = nodes
        mish = make_node(
            "Mish",
            inputs=[softplus.input[0]],
            outputs=[mul.output[0]],
            name=f"{softplus.name}/Mish",
        )
        self += [mish]
        self -= [softplus, tanh, mul]
