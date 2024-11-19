"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

import networkx as nx

from openvino2onnx.graph import OnnxGraph
from openvino2onnx.passes import L2, logger


@L2.register()
def eliminate_dead_nodes(graph: OnnxGraph):
    """Remove nodes who doesn't connect to any output"""

    dead_nodes = set()
    exit_nodes = set(
        [graph.nodes[i]["pb"].name for i in graph.nodes if graph.nodes[i]["has_output"]]
    )
    for i in nx.topological_sort(graph.reverse()):
        if not any(nx.has_path(graph, i, j) for j in exit_nodes):
            dead_nodes.add(graph.nodes[i]["pb"])
            logger.debug(f"Dead node found: {i}")
    for node in dead_nodes:
        graph.remove_onnx_node(node)
    return graph
