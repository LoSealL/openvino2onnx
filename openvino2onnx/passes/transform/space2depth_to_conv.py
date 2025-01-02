"""
Copyright Wenyi Tang 2024-2025

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

# pylint: disable=arguments-differ

from typing import List

import numpy as np
from onnx import mapping
from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from openvino2onnx.graph import OnnxGraph
from openvino2onnx.passes import PASSES
from openvino2onnx.passes.pattern import SingleNodePattern
from openvino2onnx.passes.rewriter import Rewriter
from openvino2onnx.passes.utils import make_constant


@PASSES.register(name="space2depth_to_conv")
class SpaceToDepthToConvRewriter(Rewriter):
    """Convert SpaceToDepth to Conv."""

    def __init__(self):
        super().__init__(pattern=SingleNodePattern("SpaceToDepth"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto], to_depthwise=False):
        s2d = nodes[0]
        blocksize = self.get_attribute(s2d, "blocksize")
        assert isinstance(blocksize, int)
        _, ic, _, _ = graph.tensor_shape(s2d.input[0])
        _, oc, _, _ = graph.tensor_shape(s2d.output[0])
        assert isinstance(ic, int) and isinstance(oc, int)
        dtype = graph.tensor_type(s2d.input[0])
        assert ic * blocksize**2 == oc, "invalid space2depth parameters"

        assert not to_depthwise, "not implemented yet"
        if to_depthwise:
            kernel = np.tile(
                np.eye(blocksize**2)
                .reshape([-1, blocksize, blocksize])
                .astype(mapping.TENSOR_TYPE_MAP[dtype].np_dtype),
                [ic, 1, 1],
            )[:, None]
            # TODO: output channels need to be reordered
        else:
            kernel = (
                np.tile(
                    np.eye(blocksize**2)
                    .reshape([-1, blocksize, blocksize])
                    .astype(mapping.TENSOR_TYPE_MAP[dtype].np_dtype),
                    [ic, 1, 1, 1],
                )
                .transpose([1, 0, 2, 3])
                .reshape([oc, 1, blocksize, blocksize])
            )
            # expand to [oc, ic, blocksize, blocksize]
            kernel = np.concatenate(
                [kernel, np.tile(np.zeros_like(kernel), [1, ic - 1, 1, 1])], axis=1
            )
            for i in range(1, ic):
                kernel[i::ic][:, i] = kernel[i::ic][:, 0]
                kernel[i::ic][:, 0] = 0  # swap
        conv_weight = make_constant(name=f"{s2d.name}/weight", value=kernel)
        conv = make_node(
            op_type="Conv",
            inputs=list(s2d.input) + [conv_weight.output[0]],
            outputs=s2d.output,
            name=f"{s2d.name}/conv",
            kernel_shape=[blocksize, blocksize],
            strides=[blocksize, blocksize],
            group=ic if to_depthwise else 1,
        )
        self -= s2d
        self += [conv, conv_weight]
