"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from openvino2onnx.graph import OnnxGraph
from openvino2onnx.passes import PASSES


@PASSES.register()
def eliminate_unused_input(graph: OnnxGraph) -> OnnxGraph:
    """Eliminate unused input from graph"""
    inputs = set()
    for node in graph:
        inputs.update(graph.nodes[node]["pb"].input)

    pop_list = []
    for inp in graph.inputs:
        if inp not in inputs:
            i = graph.inputs[inp]
            pop_list.append((inp, i))
    for inp, i in sorted(pop_list, key=lambda x: x[1], reverse=True):
        graph.input.pop(i)
        graph.inputs.pop(inp)

    # re-assign index of inputs.
    for i, node in enumerate(graph.input):
        graph.inputs[node.name] = i
    return graph
