"""
Copyright Wenyi Tang 2023

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

"""

from typing import Iterator

from onnx import AttributeProto

from . import Node, register


class ReduceOp(Node):
    @property
    def type_name(self) -> str:
        return self.__class__.__name__

    @property
    def attributes(self) -> Iterator[AttributeProto]:
        keepdims = AttributeProto(
            name="keepdims", type=AttributeProto.INT, i=1 if self.keep_dims else 0
        )
        if hasattr(self, "axes"):
            axes = AttributeProto(
                name="axes", type=AttributeProto.INTS, ints=list(self.axes)
            )
            return (keepdims, axes)
        return (keepdims,)


@register
class ReduceMean(ReduceOp):
    ...


@register
class ReduceMax(ReduceOp):
    ...


@register
class ReduceMin(ReduceOp):
    ...


@register
class ReduceProd(ReduceOp):
    ...


@register
class ReduceSum(ReduceOp):
    ...


@register
class ReduceSumSquare(ReduceOp):
    ...
