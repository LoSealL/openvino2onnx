"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

import numpy as np
from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from openvino2onnx.domain.intel.openvino.utils import text_to_boolean
from openvino2onnx.graph import OnnxGraph
from openvino2onnx.passes.utils import make_constant

from . import OP_CONVERT, BaseNodeConversion


@OP_CONVERT.register(name="mvn")
class MVN(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/normalization/mvn-6.html

    https://onnx.ai/onnx/operators/onnx__LayerNormalization.html
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        normalize_variance = text_to_boolean(
            self.get_attribute(ori_node, "normalize_variance")
        )
        eps = float(self.get_attribute(ori_node, "eps"))
        eps_mode = self.get_attribute(ori_node, "eps_mode").lower()

        axes = self.get_value(ori_node.input[1]).astype("int64")
        axes_norm = np.arange(axes.min(), axes.max() + 1).astype("int64")
        if len(axes) <= 1:
            rank = len(graph.tensor_shape(ori_node.input[0]))
            axes_norm = np.arange(axes.min(), rank)
        if not np.all(axes == axes_norm) and len(axes) > 1:
            raise ValueError(f"Unsupported axes: {axes}")

        axes_node = make_constant(f"{ori_node.name}/axes", axes_norm)
        r_mean = make_node(
            "ReduceMean",
            [ori_node.input[0], axes_node.output[0]],
            [f"{ori_node.name}/ReduceMean_output0"],
            name=f"{ori_node.name}/ReduceMean",
            keepdims=1,
        )

        sub = make_node(
            "Sub",
            [ori_node.input[0], r_mean.output[0]],
            [f"{ori_node.name}/Sub_output0"],
            name=f"{ori_node.name}/Sub",
        )

        if not normalize_variance:
            self += [axes_node, r_mean]
            sub.name = ori_node.name
            sub.output[0] = ori_node.output[0]
            return sub
        if eps_mode == "inside_sqrt" and np.all(axes == axes_norm):
            # NOTE: OpenVINO handles LayerNormalization differently with Onnxruntime.
            # In Onnxruntime, `scale` must match the size of normalized layer size.
            # For example, a tensor [1, 32, 8, 8] with `axis=2`, its layer size is 64,
            # so `scale` can be a shape of [64] or [8, 8] or even [1, 64].
            # While in OpenVINO, `scale` must be broadcastable to the layer size. so
            # `scale` must be a shape of [8, 8] in this case.
            axis = int(axes_norm.flatten()[0])
            scale_shape = graph.tensor_shape(ori_node.output[0])[axis:]
            scale = make_constant(
                f"{ori_node.name}/scale", np.ones(scale_shape, np.float32)
            )
            self += scale
            return make_node(
                "LayerNormalization",
                inputs=[ori_node.input[0], scale.output[0]],
                outputs=[ori_node.output[0]],
                name=ori_node.name,
                axis=axis,
                epsilon=eps,
            )
        mul = make_node(
            "Mul",
            [sub.output[0], sub.output[0]],
            [f"{ori_node.name}/Mul_output0"],
            name=f"{ori_node.name}/Mul",
        )
        r_mean2 = make_node(
            "ReduceMean",
            [mul.output[0], axes_node.output[0]],
            [f"{ori_node.name}/ReduceMean2_output0"],
            name=f"{ori_node.name}/ReduceMean2",
            keepdims=1,
        )
        sqrt = make_node(
            "Sqrt",
            [r_mean2.output[0]],
            [f"{ori_node.name}/Sqrt_output0"],
            name=f"{ori_node.name}/Sqrt",
        )
        eps_node = make_constant(f"{ori_node.name}/eps", np.array(eps, np.float32))
        add_eps = make_node(
            "Add",
            [sqrt.output[0], eps_node.output[0]],
            [f"{ori_node.name}/Add_output0"],
            name=f"{ori_node.name}/Add",
        )
        self += [axes_node, r_mean, sub, mul, r_mean2, eps_node, add_eps, sqrt]
        return make_node(
            "Div",
            inputs=[sub.output[0], add_eps.output[0]],
            outputs=[ori_node.output[0]],
            name=ori_node.name,
        )
