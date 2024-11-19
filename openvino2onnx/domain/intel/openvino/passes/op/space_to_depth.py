"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from openvino2onnx.graph import OnnxGraph

from . import OP_CONVERT, BaseNodeConversion


@OP_CONVERT.register(name="space_to_depth")
class SpaceToDepth(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/movement/space-to-depth-1.html

    https://onnx.ai/onnx/operators/onnx__SpaceToDepth.html
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        blocksize = self.get_attribute(ori_node, "block_size")
        mode = self.get_attribute(ori_node, "mode")
        if mode is not None and mode != "blocks_first":
            raise NotImplementedError(f"mode {mode} is not supported!")
        return make_node(
            "SpaceToDepth",
            inputs=ori_node.input,
            outputs=ori_node.output,
            name=ori_node.name,
            blocksize=int(blocksize),
        )
