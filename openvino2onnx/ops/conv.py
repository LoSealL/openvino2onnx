"""
Copyright Wenyi Tang 2023

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

"""

from itertools import chain

from onnx import AttributeProto

from . import Node, register


@register
class Convolution(Node):
    @classmethod
    def trans_auto_pad(cls, ir_str):
        match ir_str:
            case "explicit":
                return "NOTSET"
            case "notset":
                return "NOTSET"
            case "auto":
                return "SAME_LOWER"
            case "same_lower":
                return "SAME_LOWER"
            case "same_upper":
                return "SAME_UPPER"
            case "valid":
                return "VALID"
        raise ValueError(f"Unknown auto_pad={ir_str}")

    @property
    def type_name(self):
        return "Conv"

    @property
    def attributes(self):
        # dilations="1, 1"
        dilations = AttributeProto(
            name="dilations",
            type=AttributeProto.INTS,
            ints=map(int, self.dilations.split(",")),
        )
        # strides="1, 1"
        strides = AttributeProto(
            name="strides",
            type=AttributeProto.INTS,
            ints=map(int, self.strides.split(",")),
        )
        # pads_begin="1, 1" pads_end="1, 1"
        pads = AttributeProto(
            name="pads",
            type=AttributeProto.INTS,
            ints=chain(
                map(int, self.pads_begin.split(",")), map(int, self.pads_end.split(","))
            ),
        )
        # auto_pad="explicit"
        auto_pad = AttributeProto(
            name="auto_pad",
            type=AttributeProto.STRING,
            s=self.trans_auto_pad(self.auto_pad).encode(),
        )
        return dilations, strides, pads, auto_pad
