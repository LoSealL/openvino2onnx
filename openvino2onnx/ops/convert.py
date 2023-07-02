"""
Copyright Wenyi Tang 2023

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

"""

from onnx import AttributeProto, TensorProto

from . import Node, register


@register
class Convert(Node):
    @property
    def type_name(self):
        return "Cast"

    def to_dtype(self):
        match self.destination_type:
            case "f64":
                return TensorProto.DOUBLE
            case "f32":
                return TensorProto.FLOAT
            case "f16":
                return TensorProto.FLOAT16
            case "i64":
                return TensorProto.INT64
            case "i32":
                return TensorProto.INT32
            case "i16":
                return TensorProto.INT16
            case "i8":
                return TensorProto.INT8
            case "u64":
                return TensorProto.UINT64
            case "u32":
                return TensorProto.UINT32
            case "u16":
                return TensorProto.UINT16
            case "u8":
                return TensorProto.UINT8
        return TensorProto.UNDEFINED

    @property
    def attributes(self):
        attr_to = AttributeProto(name="to", type=AttributeProto.INT, i=self.to_dtype())
        return (attr_to,)
