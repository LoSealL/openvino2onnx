"""
Copyright Wenyi Tang 2024-2025

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from typing import Dict, List, Optional

from onnx import NodeProto, shape_inference
from onnx.tools.update_model_dims import update_inputs_outputs_dims

from openvino2onnx.graph import OnnxGraph
from openvino2onnx.passes import PASSES


@PASSES.register()
def infer_shape(
    graph: OnnxGraph,
    input_shapes: Optional[Dict[str, List[int | str]]] = None,
    output_shapes: Optional[Dict[str, List[int | str]]] = None,
) -> OnnxGraph:
    """Regenerate tensor info of graph."""
    model = graph.model
    if input_shapes and output_shapes:
        model = update_inputs_outputs_dims(
            model, input_dims=input_shapes, output_dims=output_shapes
        )
    model = shape_inference.infer_shapes(model, data_prop=True)
    graph = OnnxGraph(model)
    # FIXME This is a workaround for GroupNormalization shape inference issue.
    # Current ONNX (1.17.0) still can't infer shapes after GN.
    need_second_infer = False
    for node in graph:
        node_pb: NodeProto = graph.nodes[node]["pb"]
        if node_pb.op_type != "GroupNormalization":
            continue
        input_shape, dtype = graph.tensor_info(node_pb.input[0])
        output_shape, _ = graph.tensor_info(node_pb.output[0])
        if output_shape is None and input_shape is not None:
            graph.set_value_info(node_pb.output[0], input_shape, dtype)
            need_second_infer = True
    if need_second_infer:
        model = graph.model
        model = shape_inference.infer_shapes(model, data_prop=True)
        return OnnxGraph(model)
    return graph
