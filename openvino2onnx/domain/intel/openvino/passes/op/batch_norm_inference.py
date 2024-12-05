"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from openvino2onnx.graph import OnnxGraph

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
