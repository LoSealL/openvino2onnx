"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from typing import List

from onnx.onnx_pb import NodeProto

from openvino2onnx.graph import OnnxGraph
from openvino2onnx.passes.pattern import SingleNodePattern
from openvino2onnx.passes.rewriter import Rewriter

from . import OP_CONVERTER


class ReduceOp(Rewriter):
    """Move axes from inputs to attributes."""

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto], *args, **kwargs):
        node = nodes[0]
        axes = self.get_value(node.input[1])
        if axes is None:
            raise ValueError(f"axes of {node.name} is not constant")
        self.set_attribute(node, "axes", axes.tolist())
        noop_with_empty_axes = self.get_attribute(node, "noop_with_empty_axes")
        if noop_with_empty_axes:
            raise ValueError(
                f"{self.__class__.__name__} below opset 18 doesn't "
                "support noop_with_empty_axes=1"
            )
        node.input.pop(1)
        self += node  # refresh the edge


@OP_CONVERTER.register("ReduceMean")
class ReduceMean(ReduceOp):
    """Move axes from inputs to attributes."""

    def __init__(self):
        super().__init__(SingleNodePattern("ReduceMean"))


@OP_CONVERTER.register("ReduceMax")
class ReduceMax(ReduceOp):
    """Move axes from inputs to attributes."""

    def __init__(self):
        super().__init__(SingleNodePattern("ReduceMax"))


@OP_CONVERTER.register("ReduceMin")
class ReduceMin(ReduceOp):
    """Move axes from inputs to attributes."""

    def __init__(self):
        super().__init__(SingleNodePattern("ReduceMin"))


@OP_CONVERTER.register("ReduceProd")
class ReduceProd(ReduceOp):
    """Move axes from inputs to attributes."""

    def __init__(self):
        super().__init__(SingleNodePattern("ReduceProd"))


@OP_CONVERTER.register("ReduceL1")
class ReduceL1(ReduceOp):
    """Move axes from inputs to attributes."""

    def __init__(self):
        super().__init__(SingleNodePattern("ReduceL1"))


@OP_CONVERTER.register("ReduceL2")
class ReduceL2(ReduceOp):
    """Move axes from inputs to attributes."""

    def __init__(self):
        super().__init__(SingleNodePattern("ReduceL2"))


@OP_CONVERTER.register("ReduceLogSum")
class ReduceLogSum(ReduceOp):
    """Move axes from inputs to attributes."""

    def __init__(self):
        super().__init__(SingleNodePattern("ReduceLogSum"))


@OP_CONVERTER.register("ReduceLogSumExp")
class ReduceLogSumExp(ReduceOp):
    """Move axes from inputs to attributes."""

    def __init__(self):
        super().__init__(SingleNodePattern("ReduceLogSumExp"))


@OP_CONVERTER.register("ReduceSumSquare")
class ReduceSumSquare(ReduceOp):
    """Move axes from inputs to attributes."""

    def __init__(self):
        super().__init__(SingleNodePattern("ReduceSumSquare"))
