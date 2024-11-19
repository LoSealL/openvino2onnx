"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from openvino2onnx.graph import OnnxGraph
from openvino2onnx.passes import PASSES, logger


@PASSES.register()
def eliminate_unused_outputs(graph: OnnxGraph) -> OnnxGraph:
    """Eliminate unused input from graph"""
    for k in list(graph.outputs):
        # pylint: disable=protected-access
        if k not in graph._out_to_node:
            logger.debug(f"Removing unused output: {k}")
            graph.remove_output(k)
    return graph
