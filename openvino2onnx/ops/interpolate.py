"""
Copyright Wenyi Tang 2023

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

"""

from onnx import AttributeProto

from . import Node, register


@register
class Interpolate(Node):
    @property
    def type_name(self):
        return "Resize"

    def to_mode(self):
        if "mode" not in self:
            return "nearest"
        match self.mode:
            case "linear_onnx":
                return "linear"
        return self.mode

    @property
    def attributes(self):
        mode = AttributeProto(
            name="mode", type=AttributeProto.STRING, s=self.to_mode().encode()
        )
        coordinate_transformation_mode = AttributeProto(
            name="coordinate_transformation_mode",
            type=AttributeProto.STRING,
            s=self.coordinate_transformation_mode.encode(),
        )
        nearest_mode = AttributeProto(
            name="nearest_mode",
            type=AttributeProto.STRING,
            s=self.nearest_mode.encode(),
        )
        cubic_coeff_a = AttributeProto(
            name="cubic_coeff_a",
            type=AttributeProto.FLOAT,
            f=float(self.cube_coeff),
        )
        # do not support padding
        if "pads_begin" in self:
            assert all((int(i) == 0 for i in self.pads_begin.split(",")))
        if "pads_end" in self:
            assert all((int(i) == 0 for i in self.pads_end.split(",")))
        return mode, coordinate_transformation_mode, nearest_mode, cubic_coeff_a
