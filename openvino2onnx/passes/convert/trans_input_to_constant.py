"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

import numpy as np

from openvino2onnx.graph import OnnxGraph
from openvino2onnx.passes import PASSES
from openvino2onnx.passes.utils import make_constant


@PASSES.register()
def trans_input_to_constant(graph: OnnxGraph, input_name: str, value: np.ndarray):
    """Consolidate a input to a fixed value as a constant node.

    Args:
        graph (OnnxGraph): _description_
        input_name (str): _description_
        value (np.ndarray):
    """

    if input_name not in graph.inputs:
        raise ValueError(f"{input_name} is not an input of the model")

    node = make_constant(input_name + "/Const", value)
    node.output[0] = input_name
    graph.add_onnx_node(node)
    return graph
