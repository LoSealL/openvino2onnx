"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

import onnx
from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from openvino2onnx.graph import OnnxGraph

from . import OP_CONVERT, BaseNodeConversion


@OP_CONVERT.register(name="convert")
class Convert(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/type/convert-1.html

    https://onnx.ai/onnx/operators/onnx__Cast.html
    """

    def _to_dtype(self, destination_type: str):
        match destination_type:
            case "f64":
                return onnx.TensorProto.DOUBLE
            case "f32":
                return onnx.TensorProto.FLOAT
            case "f16":
                return onnx.TensorProto.FLOAT16
            case "i64":
                return onnx.TensorProto.INT64
            case "i32":
                return onnx.TensorProto.INT32
            case "i16":
                return onnx.TensorProto.INT16
            case "i8":
                return onnx.TensorProto.INT8
            case "u64":
                return onnx.TensorProto.UINT64
            case "u32":
                return onnx.TensorProto.UINT32
            case "u16":
                return onnx.TensorProto.UINT16
            case "u8":
                return onnx.TensorProto.UINT8
        return onnx.TensorProto.UNDEFINED

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        destination_type = self.get_attribute(ori_node, "destination_type")
        return make_node(
            "Cast",
            inputs=ori_node.input,
            outputs=ori_node.output,
            name=ori_node.name,
            to=self._to_dtype(destination_type),
        )
