"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

:Author: Jianjin Liao
:Email: jianjin.liao@intel.com
"""

# pylint: disable=arguments-differ

from typing import List

import numpy as np
from onnx.onnx_pb import NodeProto

from ... import OnnxGraph
from .. import PASSES
from ..pattern import SingleNodePattern
from ..rewriter import Rewriter
from ..utils import make_constant


@PASSES.register(name="shape_to_constant")
class ShapeToConstantPass(Rewriter):
    """Convert static Shape op output to Constant."""

    def __init__(self):
        super().__init__(pattern=SingleNodePattern("Shape"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        node = nodes[0]
        try:
            shape = graph.static_tensor_shape(node.input[0])
        except ValueError:
            # shape is not constant
            return
        # replace Shape with Constant
        shape_const = make_constant(node.name + "/Reshape", np.array(shape, "int64"))
        shape_const.output[0] = node.output[0]
        self -= node
        self += shape_const
