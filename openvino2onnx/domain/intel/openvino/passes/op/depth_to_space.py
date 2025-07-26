"""
Copyright (C) 2024 The OPENVINO2ONNX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from .. import OnnxGraph
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
