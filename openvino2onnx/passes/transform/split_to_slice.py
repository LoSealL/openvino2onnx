"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

# pylint: disable=arguments-differ

from typing import List

import numpy as np
from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from openvino2onnx.graph import OnnxGraph
from openvino2onnx.passes import L2
from openvino2onnx.passes.pattern import SingleNodePattern
from openvino2onnx.passes.rewriter import Rewriter
from openvino2onnx.passes.utils import make_constant


@L2.register(name="split_to_slice", deps=["initializer_to_constant"])
class SplitToSliceRewriter(Rewriter):
    """Change Split node to Slice node."""

    def __init__(self):
        super().__init__(SingleNodePattern("Split"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        node = nodes[0]
        axis = self.get_attribute(node, "axis") or 0
        assert isinstance(axis, int)
        if len(node.input) < 2:
            num_outputs = self.get_attribute(node, "num_outputs")
            if num_outputs is None:
                num_outputs = graph.tensor_shape(node.input[0])[axis]
            assert isinstance(num_outputs, int)
            shape = graph.tensor_shape(node.input[0])[axis]
            assert isinstance(shape, int)
            if shape % num_outputs == 0:
                split = [shape // num_outputs] * num_outputs
            else:
                remind = shape % num_outputs
                split = [(shape - remind) // (num_outputs - 1)] * (num_outputs - 1)
                split.append(remind)
        else:
            split = self.get_input_node(node, 1)
            assert split is not None
            split = self.get_value(split)
            assert split is not None, "split input should be constant"
        starts = 0
        for i, (ch, _) in enumerate(zip(split, node.output)):
            starts_node = make_constant(
                name=f"{node.name}/Starts{i}", value=np.array([starts], "int64")
            )
            ends_node = make_constant(
                name=f"{node.name}/Ends{i}", value=np.array([starts + ch], "int64")
            )
            axes_node = make_constant(
                name=f"{node.name}/Axes{i}", value=np.array([axis], "int64")
            )
            slice_node = make_node(
                op_type="Slice",
                inputs=[
                    node.input[0],
                    starts_node.output[0],
                    ends_node.output[0],
                    axes_node.output[0],
                ],
                outputs=[node.output[i]],
                name=f"{node.name}/Slice{i}",
            )
            self += [starts_node, ends_node, axes_node, slice_node]
            starts += ch
        self -= node
