"""
Copyright Wenyi Tang 2023

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

"""

from itertools import chain

import numpy as np
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

    @classmethod
    def convert_ceil_to_floor(cls, in_shape, strides, dilations, kernel_size, paddings):
        """Convert pool ceil_mode "CEIL" to "FLOOR"."""
        h, w = in_shape[-2:]
        pad_begin = [paddings[1], paddings[0]]
        pad_end = [paddings[3], paddings[2]]
        out_paddings = []
        for x, p0, p1, k, d, s in zip(
            (h, w), pad_begin, pad_end, kernel_size, dilations, strides
        ):
            out = (x + p0 + p1 - ((k - 1) * d + 1)) / s + 1
            if np.ceil(out) > np.floor(out):
                out_paddings.append(s - 1)
            else:
                out_paddings.append(p1)
        return [*paddings[:2], *reversed(out_paddings)]

    @property
    def type_name(self):
        return "MaxPool"

    @property
    def attributes(self):
        _ceil_mode = 0
        if hasattr(self, "rounding_type"):
            _ceil_mode = self.trans_ceil_mode(self.rounding_type)
        _shape = list(map(int, self.inputs["0"]["dim"]))
        _paddings = list(
            chain(
                map(int, self.pads_begin.split(",")), map(int, self.pads_end.split(","))
            )
        )
        _strides = list(map(int, self.strides.split(",")))
        if hasattr(self, "dilations"):
            _dilations = list(map(int, map(float, self.dilations.split(","))))
        else:
            _dilations = [1, 1]
        _kernel = list(map(int, self.kernel.split(",")))
        if _ceil_mode == 1:
            _paddings = self.convert_ceil_to_floor(
                _shape, _strides, _dilations, _kernel, _paddings
            )
        # rounding_type="floor"
        ceil_mode = AttributeProto(
            name="ceil_mode",
            type=AttributeProto.INT,
            i=0,
        )
        # dilations="1, 1" in opset 19
        dilations = AttributeProto(  # noqa
            name="dilations",
            type=AttributeProto.INTS,
            ints=_dilations,
        )
        # pads_begin="1, 1" pads_end="1, 1"
        pads = AttributeProto(
            name="pads",
            type=AttributeProto.INTS,
            ints=_paddings,
        )
        # strides="1, 1"
        strides = AttributeProto(
            name="strides",
            type=AttributeProto.INTS,
            ints=_strides,
        )
        # kernel="3, 3"
        kernel_shape = AttributeProto(
            name="kernel_shape",
            type=AttributeProto.INTS,
            ints=_kernel,
        )
        return ceil_mode, pads, strides, kernel_shape


@register
class AvgPool(MaxPool):
    @property
    def type_name(self):
        return "AveragePool"
