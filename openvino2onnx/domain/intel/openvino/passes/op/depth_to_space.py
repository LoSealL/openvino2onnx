"""
Copyright Wenyi Tang 2024-2025

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from openvino2onnx.graph import OnnxGraph

from . import OP_CONVERT, BaseNodeConversion


@OP_CONVERT.register(name="depth_to_space")
class DepthToSpace(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/movement/depth-to-space-1.html

    https://onnx.ai/onnx/operators/onnx__DepthToSpace.html
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        blocksize = self.get_attribute(ori_node, "block_size")
        assert isinstance(blocksize, (int, float, str))
        mode = self.get_attribute(ori_node, "mode")
        return make_node(
            "DepthToSpace",
            inputs=ori_node.input,
            outputs=ori_node.output,
            name=ori_node.name,
            blocksize=int(blocksize),
            mode="DCR" if mode == "blocks_first" else "CRD",
        )
