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


@OP_CONVERT.register(name="batch_norm_inference")
class BatchNormInference(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/normalization/batch-norm-inference-5.html

    https://onnx.ai/onnx/operators/onnx__BatchNormalization.html
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        epsilon = self.get_attribute(ori_node, "epsilon")
        if isinstance(epsilon, str):
            epsilon = float(epsilon)
        if not isinstance(epsilon, float):
            epsilon = 1e-5

        return make_node(
            "BatchNormalization",
            inputs=ori_node.input,
            outputs=ori_node.output,
            name=ori_node.name,
            epsilon=epsilon,
        )
