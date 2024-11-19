"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

import networkx as nx

from openvino2onnx.graph import OnnxGraph
from openvino2onnx.passes import PASSES, logger

from . import OP_CONVERTER


def _expand_deps(deps):
    root = nx.DiGraph()
    root.add_nodes_from(deps)
    leaves: list = deps.copy()
    while leaves:
        leaf = leaves.pop(0)
        children = PASSES.get(leaf).__deps__
        leaves.extend(children)
        root.add_edges_from([(leaf, child) for child in children])
        try:
            cycles = nx.find_cycle(root, leaf)
        except nx.NetworkXNoCycle:
            continue
        else:
            logger.error(f"Cyclic dependencies found!: {cycles}")
            break
    yield from reversed(list(nx.topological_sort(root)))


def downgrade_op_version(graph: OnnxGraph, op_version: int = 17):
    """Downgrade the op version of all nodes in the graph to the specified version."""

    if op_version not in (19, 17, 13):
        raise NotImplementedError(
            f"Support to downgrade to 13, 17 or 19 for now, got {op_version}."
        )
    if graph.opset_version <= op_version:
        return graph
    if graph.opset_version > 19 and op_version <= 17:
        graph = downgrade_op_version(graph, 19)
        return downgrade_op_version(graph, op_version)
    if graph.opset_version > 17 and op_version <= 13:
        graph = downgrade_op_version(graph, 17)
        return downgrade_op_version(graph, op_version)

    logger.debug(f"downgrading opset to version {op_version}")
    node_types = set(node["pb"].op_type for node in graph.nodes.values())
    for node_type in node_types:
        if converter := OP_CONVERTER[op_version].get(node_type):
            try:
                for deps in _expand_deps(converter.__deps__):
                    graph = PASSES.get(deps)(graph)
                graph = converter(graph)
            except Exception:  # pylint: disable=broad-except
                logger.error(f"Failed to downgrade {node_type} to {op_version}")
                raise
    graph.opset_version = op_version
    return graph
