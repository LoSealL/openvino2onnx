"""
Copyright Wenyi Tang 2024-2025

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from onnx.helper import make_node, make_tensor_type_proto, make_value_info

from openvino2onnx.graph import OnnxGraph
from openvino2onnx.passes import PASSES


@PASSES.register(deps=["infer_shape"])
def reorder_to_nhwc(
    graph: OnnxGraph, input_names: Optional[List[str]] = None
) -> OnnxGraph:
    """Reorder input tensor to NHWC format by inserting a dummy transpose op.

    Args:
        graph (OnnxGraph): the graph to rewrite
        input_names (List[str], optional): the names of input tensors to reorder.
            Defaults to None, which means all input tensors will be reordered.

    Returns:
        OnnxGraph: the rewritten graph
    """

    input_feeds: Dict[str, Tuple[List[int | str] | None, int]] = {}
    for input_name in graph.inputs:
        input_shape = graph.tensor_shape(input_name)
        if len(input_shape) == 4:
            # this pass only works for NCHW 4D tensor
            if not input_names or input_name in input_names:
                input_feeds[input_name] = graph.tensor_info(input_name)
    if not input_feeds:
        return graph

    for input_name, (input_shape, dtype) in input_feeds.items():
        assert input_shape is not None
        permute = make_node(
            "Transpose",
            inputs=[f"{input_name}_nhwc"],
            outputs=[input_name],
            name=f"nchw_transpose/{input_name}",
            perm=[0, 3, 1, 2],
        )
        graph.input.append(
            make_value_info(
                f"{input_name}_nhwc",
                make_tensor_type_proto(
                    dtype, np.take(input_shape, [0, 2, 3, 1]).tolist()
                ),
            )
        )
        graph.inputs[f"{input_name}_nhwc"] = len(graph.input) - 1
        graph.add_onnx_node(permute)
    return graph
