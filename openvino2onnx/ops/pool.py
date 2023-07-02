"""
Copyright Wenyi Tang 2023

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

"""

from itertools import chain

from onnx import AttributeProto

from . import Node, register


@register
class MaxPool(Node):
    @classmethod
    def trans_ceil_mode(cls, ir_mode):
        match ir_mode:
            case "ceil":
                return 1
        return 0

    @property
    def type_name(self):
        return "MaxPool"

    @property
    def attributes(self):
        # rounding_type="ceil"
        ceil_mode = AttributeProto(
            name="ceil_mode",
            type=AttributeProto.INT,
            i=self.trans_ceil_mode(self.rounding_type),
        )
        # dilations="1, 1"
        dilations = AttributeProto(
            name="dilations",
            type=AttributeProto.INTS,
            ints=map(int, self.dilations.split(",")),
        )
        # pads_begin="1, 1" pads_end="1, 1"
        pads = AttributeProto(
            name="pads",
            type=AttributeProto.INTS,
            ints=chain(
                map(int, self.pads_begin.split(",")), map(int, self.pads_end.split(","))
            ),
        )
        # strides="1, 1"
        strides = AttributeProto(
            name="strides",
            type=AttributeProto.INTS,
            ints=map(int, self.strides.split(",")),
        )
        # kernel="3, 3"
        kernel_shape = AttributeProto(
            name="kernel_shape",
            type=AttributeProto.INTS,
            ints=map(int, self.kernel.split(",")),
        )
        return ceil_mode, dilations, pads, strides, kernel_shape
