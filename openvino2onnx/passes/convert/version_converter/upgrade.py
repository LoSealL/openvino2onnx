"""
Copyright Wenyi Tang 2024-2025

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

import onnx.version
from onnx.version_converter import convert_version

from openvino2onnx.graph import OnnxGraph
from openvino2onnx.passes import logger


def upgrade_op_version(graph: OnnxGraph, op_version: int = 17):
    """Upgrade the op version of all nodes in the graph to the specified version."""

    if graph.opset_version >= op_version:
        return graph

    logger.debug(f"Upgrading opset version from {graph.opset_version} to {op_version}")
    if graph.functions:
        logger.warning(
            "upgrade onnx version will lose functions in the graph, "
            f"this is a bug of onnx {onnx.version.version}. "
            f"Add -v={graph.opset_version} to avoid this auto upgradation."
        )
    new_model = convert_version(graph.model, op_version)
    return OnnxGraph(new_model)
