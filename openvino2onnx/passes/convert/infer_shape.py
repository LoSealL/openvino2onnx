"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from onnx import shape_inference

from openvino2onnx.graph import OnnxGraph
from openvino2onnx.passes import PASSES


@PASSES.register()
def infer_shape(graph: OnnxGraph) -> OnnxGraph:
    """Regenerate tensor info of graph."""
    model = shape_inference.infer_shapes(graph.model, data_prop=True)
    return OnnxGraph(model)
