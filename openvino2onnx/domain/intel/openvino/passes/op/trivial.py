"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from openvino2onnx.graph import OnnxGraph
from openvino2onnx.passes.pattern import SingleNodePattern

from . import OP_CONVERT, BaseNodeConversion


@OP_CONVERT.register(name="z__trivial")  # make sure to be called in the last
class Trivial(BaseNodeConversion):
    """Trivial conversion for OpenVINO nodes."""

    def __init__(self):
        super().__init__(SingleNodePattern().with_attr("version"))

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        # replace ai.intel.openvino domain to ai.onnx (empty)
        if ori_node.domain != "ai.intel.openvino":
            raise RuntimeError(
                f"node {ori_node.name} got an unexpected domain: {ori_node.domain}"
            )
        return make_node(
            ori_node.op_type,
            inputs=ori_node.input,
            outputs=ori_node.output,
            name=ori_node.name,
        )
