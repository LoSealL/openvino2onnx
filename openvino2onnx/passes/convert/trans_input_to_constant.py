"""
Copyright (C) 2025 The OPENVINO2ONNX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np

from ... import OnnxGraph
from .. import PASSES
from ..utils import make_constant


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
