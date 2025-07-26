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

# pylint: disable=arguments-differ
from copy import deepcopy
from typing import List

from onnx.onnx_pb import NodeProto

from ...graph import OnnxGraph
from .. import PASSES
from ..pattern import GraphPattern, SingleNodePattern
from ..rewriter import Rewriter


@PASSES.register(name="reorder_quantizelinear")
class ReorderQuantizeLinearRewriter(Rewriter):
    """Reorder X - Q - DQ pattern to Q - X - DQ where X match the following pattern:

    - `X` has no parameter variables (such as Conv, MatMul, etc.)
    - `X` doesn't change channels when Q-DQ is channel-wise

    Example::

        Before:

            SpaceToDepth - QuantizeLinear - DequantizeLinear

        After:

            QuantizeLinear - SpaceToDepth - DequantizeLinear
    """

    _ALLOW_X_TYPES = {"SpaceToDepth", "DepthToSpace", "MaxPool", "AveragePool"}

    def __init__(self):
        pattern = GraphPattern()
        p1 = SingleNodePattern("QuantizeLinear")
        p2 = SingleNodePattern("DequantizeLinear")
        pattern.add_edge(p1, p2)
        super().__init__(pattern)

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        qnode, dqnode = nodes
        assert qnode.op_type == "QuantizeLinear"
        assert dqnode.op_type == "DequantizeLinear"

        upstream_nodes = graph.onnx_predecessors(qnode)
        if len(upstream_nodes) != 1:
            return  # not a single path

        xnode = upstream_nodes[0]
        if xnode.op_type not in self._ALLOW_X_TYPES:
            return  # not a valid X node

        qnode_new = deepcopy(qnode)
        qnode_new.name += f"/{self.__class__.__name__}"
        dqnode_new = deepcopy(dqnode)
        dqnode_new.name += f"/{self.__class__.__name__}"
        xnode_new = deepcopy(xnode)
        xnode_new.name += f"/{self.__class__.__name__}"

        # swap X input and Q input
        qnode_new.input[0], xnode_new.input[0] = xnode.input[0], qnode.output[0]
        dqnode_new.input[0] = xnode.output[0]

        self -= [xnode, qnode, dqnode]
        self += [xnode_new, qnode_new, dqnode_new]
