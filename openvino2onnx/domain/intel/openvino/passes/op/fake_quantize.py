"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

import numpy as np
from onnx.helper import make_node
from onnx.mapping import TENSOR_TYPE_MAP
from onnx.onnx_pb import NodeProto

from openvino2onnx.graph import OnnxGraph
from openvino2onnx.passes.utils import make_constant

from . import OP_CONVERT, BaseNodeConversion


@OP_CONVERT.register(name="fake_quantize")
class FakeQuantize(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/quantization/fake-quantize-1.html

    Split fake quantize to QuantizeLinear and DequantizeLinear.
    https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html
    https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html

    Note:

        FakeQuantize only applies to activations, not weights.
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        levels = int(self.get_attribute(ori_node, "levels") or 256)
        nbits = int(np.log2(levels))
        assert nbits in (8, 16, 32)
        input_low = self.get_value(ori_node.input[1]).squeeze()
        input_high = self.get_value(ori_node.input[2]).squeeze()
        output_low = self.get_value(ori_node.input[3]).squeeze()
        output_high = self.get_value(ori_node.input[4]).squeeze()

        try:
            scale_prec = graph.tensor_type(ori_node.output[0])
            scale_prec = TENSOR_TYPE_MAP[scale_prec].np_dtype
        except ValueError:
            scale_prec = np.float32
        zero_prec = np.iinfo(f"uint{nbits}").dtype
        scales = ((input_high - input_low) / (levels - 1)).astype(scale_prec)
        zero_points = np.rint((levels - 1) * input_low / (input_low - input_high))
        zero_points = zero_points.astype(zero_prec)
        scale_node = make_constant(f"{ori_node.name}/QuantizeLinear_y_scale", scales)
        zp_node = make_constant(
            f"{ori_node.name}/QuantizeLinear_y_zero_point", zero_points
        )
        # add QuantizeLinear
        quantizelinear = make_node(
            "QuantizeLinear",
            inputs=[ori_node.input[0], scale_node.output[0], zp_node.output[0]],
            outputs=[f"{ori_node.name}/QuantizeLinear_output0"],
            name=f"{ori_node.name}/QuantizeLinear",
        )
        self += [scale_node, zp_node, quantizelinear]
        # replace self with DequantizeLinear
        scales = ((output_high - output_low) / (levels - 1)).astype(scale_prec)
        zero_points = np.rint((levels - 1) * output_low / (output_low - output_high))
        zero_points = zero_points.astype(zero_prec)
        scale_node = make_constant(f"{ori_node.name}/DequantizeLinear_y_scale", scales)
        zp_node = make_constant(
            f"{ori_node.name}/DequantizeLinear_y_zero_point", zero_points
        )
        self += [scale_node, zp_node]
        return make_node(
            "DequantizeLinear",
            inputs=[quantizelinear.output[0], scale_node.output[0], zp_node.output[0]],
            outputs=ori_node.output,
            name=ori_node.name,
        )
