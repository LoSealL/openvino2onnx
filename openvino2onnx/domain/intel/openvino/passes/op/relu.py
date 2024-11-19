"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from openvino2onnx import OPENVINO2ONNX_OPSET
from openvino2onnx.graph import OnnxGraph

from . import OP_CONVERT, BaseNodeConversion


@OP_CONVERT.register(name="prelu")
class PReLU(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/activation/prelu-1.html

    https://onnx.ai/onnx/operators/onnx__PRelu.html
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        input_type = graph.tensor_type(ori_node.input[0])
        slope_type = graph.tensor_type(ori_node.input[1])
        if input_type != slope_type:
            cast = make_node(
                "Cast",
                inputs=[ori_node.input[1]],
                outputs=[f"{ori_node.input[1]}_cast"],
                name=f"{ori_node.name}/Cast",
                to=input_type,
            )
            ori_node.input[1] = cast.output[0]
            self += cast
        return make_node(
            "PRelu", inputs=ori_node.input, outputs=ori_node.output, name=ori_node.name
        )


@OP_CONVERT.register(name="relu")
class ReLU(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/activation/relu-1.html

    https://onnx.ai/onnx/operators/onnx__Relu.html
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        return make_node(
            "Relu", inputs=ori_node.input, outputs=ori_node.output, name=ori_node.name
        )


@OP_CONVERT.register(name="gelu")
class Gelu(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/activation/gelu-7.html

    https://onnx.ai/onnx/operators/onnx__Gelu.html
    """

    def __init__(self):
        super().__init__()
        self.register_post_hook(self.post_hook)

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        approximate = self.get_attribute(ori_node, "approximation_mode").lower()
        return make_node(
            "Gelu",
            inputs=ori_node.input,
            outputs=ori_node.output,
            name=ori_node.name,
            approximate="tanh" if approximate == "tanh" else "none",
        )

    def post_hook(self, graph: OnnxGraph):
        """Convert Gelu to subgraph for opset below 20."""
        # pylint: disable=import-outside-toplevel
        from openvino2onnx.passes.convert.version_converter.opset19.gelu import (
            Gelu as hook,
        )

        if OPENVINO2ONNX_OPSET.version < 20:
            return hook()(graph)
        return graph


@OP_CONVERT.register(name="selu")
class Selu(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/activation/selu-1.html

    https://onnx.ai/onnx/operators/onnx__Selu.html
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        alpha = float(self.get_value(ori_node.input[1]).squeeze())
        gamma = float(self.get_value(ori_node.input[2]).squeeze())
        return make_node(
            "Selu",
            inputs=[ori_node.input[0]],
            outputs=ori_node.output,
            name=ori_node.name,
            alpha=alpha,
            gamma=gamma,
        )
