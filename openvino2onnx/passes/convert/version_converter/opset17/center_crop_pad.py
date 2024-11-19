"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from typing import List

import numpy as np
from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from openvino2onnx.graph import OnnxGraph
from openvino2onnx.passes.pattern import SingleNodePattern
from openvino2onnx.passes.rewriter import Rewriter
from openvino2onnx.passes.utils import make_constant

from . import OP_CONVERTER


@OP_CONVERTER.register("CenterCropPad")
class CenterCropPad(Rewriter):
    """CenterCropPad to pad and slice"""

    def __init__(self):
        super().__init__(SingleNodePattern("CenterCropPad"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto], *args, **kwargs):
        node = nodes[0]
        axes = self.get_attribute(node, "axes")
        shape = self.get_value(node.input[1])
        if shape is None:
            raise RuntimeError(f"shape of {node.name} is not constant")
        ori_shape = graph.tensor_shape(node.input[0])
        if axes is None:
            axes = list(range(len(shape)))
        axes = sorted(axes)

        if self._need_pad(shape, ori_shape, axes):
            shape_iter = iter(shape)
            shape_expand = list(ori_shape)
            for i in axes:
                shape_expand[i] = next(shape_iter)
            paddings = [max(0, i - j) for i, j in zip(shape_expand, ori_shape)]
            paddings = np.array([[i // 2, i - i // 2] for i in paddings], np.int64)
            paddings = paddings.transpose().flatten()
            pad_cst = make_constant(f"{node.name}/Pad/pads", paddings)
            pad = make_node(
                "Pad",
                inputs=[node.input[0], pad_cst.output[0]],
                outputs=[f"{node.name}/Pad_output0"],
                name=f"{node.name}/Pad",
            )
            self += [pad_cst, pad]
            node.input[0] = pad.output[0]
            ori_shape = [max(i, j) for i, j in zip(shape_expand, ori_shape)]
        # get slice begin and end
        begin, end = self._slice_begin_and_end(shape, ori_shape, axes)
        beg_node = make_constant(f"{node.name}/Slice/begin", np.array(begin, np.int64))
        end_node = make_constant(f"{node.name}/Slice/end", np.array(end, np.int64))
        axes_node = make_constant(f"{node.name}/Slice/axes", np.array(axes, np.int64))
        slice_node = make_node(
            "Slice",
            inputs=[
                node.input[0],
                beg_node.output[0],
                end_node.output[0],
                axes_node.output[0],
            ],
            outputs=node.output,
            name=node.name,
        )
        self += [beg_node, end_node, axes_node, slice_node]
        self -= node

    def _need_pad(self, shape, ori_shape, axes):
        shape_iter = iter(shape)
        return any(next(shape_iter) > ori_shape[i] for i in axes)

    def _slice_begin_and_end(self, shape, ori_shape, axes):
        shape_iter = iter(shape)
        slices = [ori_shape[i] - next(shape_iter) for i in axes]
        begin = [s // 2 for s in slices]
        end = [ori_shape[i] - s // 2 for i, s in zip(axes, slices)]
        return begin, end
