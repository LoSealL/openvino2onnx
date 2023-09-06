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
    def keepdims(self):
        """convert keep_dims string to boolean"""
        if isinstance(self.keep_dims, str):
            return self.keep_dims.lower() == "true"
        return self.keep_dims

    @property
    def attributes(self) -> Iterator[AttributeProto]:
        keepdims = AttributeProto(
            name="keepdims", type=AttributeProto.INT, i=1 if self.keepdims else 0
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
