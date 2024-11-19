"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

import numpy as np
from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from openvino2onnx.graph import OnnxGraph
from openvino2onnx.passes.utils import make_constant

from . import OP_CONVERT, BaseNodeConversion


@OP_CONVERT.register(name="strided_slice")
class StridedSlice(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/movement/strided-slice-1.html

    https://onnx.ai/onnx/operators/onnx__Slice.html
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        # move steps to input[4]
        if len(ori_node.input) > 4:
            ori_node.input[4] = ori_node.input[3]
        else:
            ori_node.input.append(ori_node.input[3])
        ori_node.input[3] = ""
        begin_mask = self.get_attribute(ori_node, "begin_mask")
        end_mask = self.get_attribute(ori_node, "end_mask")
        begin_mask = list(map(int, begin_mask.split(",")))
        end_mask = list(map(int, end_mask.split(",")))
        shrink_axis_mask = self.get_attribute(ori_node, "shrink_axis_mask")

        if any(i != 0 for i in begin_mask + end_mask):
            # adjust begin and end
            data_shape = graph.tensor_shape(ori_node.input[0])
            begin_var = self.get_value(ori_node.input[1]).copy()
            end_var = self.get_value(ori_node.input[2]).copy()
            for i, x in enumerate(begin_mask):
                begin_var[i] = 0 if x != 0 else begin_var[i]
            for i, x in enumerate(end_mask):
                end_var[i] = data_shape[i] if x != 0 else end_var[i]
            starts = make_constant(f"{ori_node.name}/starts", begin_var)
            ends = make_constant(f"{ori_node.name}/ends", end_var)
            ori_node.input[1] = starts.output[0]
            ori_node.input[2] = ends.output[0]
            self += [starts, ends]
        if not shrink_axis_mask:
            shrink_axis_mask = []
        else:
            shrink_axis_mask = list(map(int, shrink_axis_mask.split(",")))
        if any(i != 0 for i in shrink_axis_mask):
            # add a squeeze after slice
            (axes,) = np.nonzero(shrink_axis_mask)
            axes_node = make_constant(
                f"{ori_node.name}/Squeeze/axes", np.array(axes, dtype=np.int64)
            )
            squeeze = make_node(
                "Squeeze",
                name=f"{ori_node.name}/Squeeze",
                inputs=[f"{ori_node.name}/Slice_output0", axes_node.output[0]],
                outputs=[ori_node.output[0]],
            )
            ori_node.output[0] = f"{ori_node.name}/Slice_output0"
            self += [squeeze, axes_node]

        return make_node(
            "Slice",
            inputs=ori_node.input,
            outputs=ori_node.output,
            name=ori_node.name,
        )
