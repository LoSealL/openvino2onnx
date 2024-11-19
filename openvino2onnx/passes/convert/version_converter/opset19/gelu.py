"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from typing import List

import numpy as np
from onnx.helper import make_node, tensor_dtype_to_np_dtype
from onnx.onnx_pb import NodeProto

from openvino2onnx.graph import OnnxGraph
from openvino2onnx.passes.pattern import SingleNodePattern
from openvino2onnx.passes.rewriter import Rewriter
from openvino2onnx.passes.utils import make_constant

from . import OP_CONVERTER


@OP_CONVERTER.register("Gelu")
class Gelu(Rewriter):
    """Rewrite Gelu using Erf.

    Gelu(x) = 0.5 * x * (1 + Erf(x / sqrt(2)))

    or

    Gelu(x) = 0.5 * x * (1 + Tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))

    If approximate is set to "tanh".
    """

    def __init__(self):
        super().__init__(SingleNodePattern("Gelu"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto], *args, **kwargs):
        node = nodes[0]
        appr = self.get_attribute(node, "approximate")
        if appr == "tanh":
            erf_node = self.rewrite_tanh(graph, node)
        else:
            erf_node = self.rewrite_erf(graph, node)
        dtype = graph.tensor_type(node.input[0])
        dtype = tensor_dtype_to_np_dtype(dtype)
        zero_dot_five_cst = make_constant(
            f"{node.name}/zero_dot_five", np.array(0.5, dtype)
        )
        mul_05 = make_node(
            "Mul",
            inputs=[node.input[0], zero_dot_five_cst.output[0]],
            outputs=[f"{node.name}/Mul_05_output0"],
            name=f"{node.name}/Mul_05",
        )
        mul = make_node(
            "Mul",
            inputs=[mul_05.output[0], erf_node.output[0]],
            outputs=node.output,
            name=node.name,
        )
        self -= node
        self += [zero_dot_five_cst, mul_05, mul]

    def rewrite_tanh(self, graph: OnnxGraph, node: NodeProto):
        dtype = graph.tensor_type(node.input[0])
        dtype = tensor_dtype_to_np_dtype(dtype)
        pow3_cst = make_constant(f"{node.name}/pow3", np.array(3.0, dtype))
        pow3 = make_node(
            "Pow",
            inputs=[node.input[0], pow3_cst.output[0]],
            outputs=[f"{node.name}/Pow_3_output0"],
            name=f"{node.name}/Pow_3",
        )
        pow3_coeff_cst = make_constant(
            f"{node.name}/pow3_coeff", np.array(0.044715, dtype)
        )
        mul_coeff = make_node(
            "Mul",
            inputs=[pow3.output[0], pow3_coeff_cst.output[0]],
            outputs=[f"{node.name}/Mul_coeff_output0"],
            name=f"{node.name}/Mul_coeff",
        )
        add_pow3 = make_node(
            "Add",
            inputs=[node.input[0], mul_coeff.output[0]],
            outputs=[f"{node.name}/Add_pow3_output0"],
            name=f"{node.name}/Add_pow3",
        )
        sqrt2pi_cst = make_constant(
            f"{node.name}/sqrt2pi", np.sqrt(2.0 / np.pi).astype(dtype)
        )
        mul_sqrt2pi = make_node(
            "Mul",
            inputs=[sqrt2pi_cst.output[0], add_pow3.output[0]],
            outputs=[f"{node.name}/Mul_sqrt2pi_output0"],
            name=f"{node.name}/Mul_sqrt2pi",
        )
        tanh = make_node(
            "Tanh",
            inputs=[mul_sqrt2pi.output[0]],
            outputs=[f"{node.name}/Tanh_output0"],
            name=f"{node.name}/Tanh",
        )
        plus1_cst = make_constant(f"{node.name}/plus1", np.array(1.0, dtype))
        tanh_plus1 = make_node(
            "Add",
            inputs=[tanh.output[0], plus1_cst.output[0]],
            outputs=[f"{node.name}/Add_tanh_plus1_output0"],
            name=f"{node.name}/Add_tanh_plus1",
        )
        self += [
            pow3_cst,
            pow3,
            pow3_coeff_cst,
            mul_coeff,
            add_pow3,
            sqrt2pi_cst,
            mul_sqrt2pi,
            tanh,
            plus1_cst,
            tanh_plus1,
        ]
        return tanh_plus1

    def rewrite_erf(self, graph: OnnxGraph, node: NodeProto):
        dtype = graph.tensor_type(node.input[0])
        dtype = tensor_dtype_to_np_dtype(dtype)
        sqrt2_cst = make_constant(f"{node.name}/sqrt2", np.sqrt(2.0).astype(dtype))
        div_sqrt2 = make_node(
            "Div",
            inputs=[node.input[0], sqrt2_cst.output[0]],
            outputs=[f"{node.name}/Div_sqrt2_output0"],
            name=f"{node.name}/Div_sqrt2",
        )
        erf = make_node(
            "Erf",
            inputs=[div_sqrt2.output[0]],
            outputs=[f"{node.name}/Erf_output0"],
            name=f"{node.name}/Erf",
        )
        plus1_cst = make_constant(f"{node.name}/plus1", np.array(1.0, dtype))
        erf_plus1 = make_node(
            "Add",
            inputs=[erf.output[0], plus1_cst.output[0]],
            outputs=[f"{node.name}/Add_erf_plus1_output0"],
            name=f"{node.name}/Add_erf_plus1",
        )
        self += [sqrt2_cst, div_sqrt2, erf, plus1_cst, erf_plus1]
        return erf_plus1
