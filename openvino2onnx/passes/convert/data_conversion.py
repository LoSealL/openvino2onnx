"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

# pylint: disable=arguments-differ

import itertools

from onnx import TensorProto
from onnx.helper import make_attribute
from onnx.numpy_helper import from_array, to_array

from openvino2onnx.graph import OnnxGraph
from openvino2onnx.passes import L1, PASSES


@L1.register()
def half_to_float(graph: OnnxGraph) -> OnnxGraph:
    """Convert half consts and values to float32."""

    for node in graph:
        node_pb = graph.nodes[node]["pb"]
        if node_pb.op_type == "Constant":
            tensor = node_pb.attribute[0].t
            if tensor.data_type == TensorProto.FLOAT16:
                array = to_array(tensor).astype("float32")
                attr = make_attribute(key="value", value=from_array(array))
                node_pb.attribute.pop()
                node_pb.attribute.append(attr)
        elif node_pb.op_type == "Cast":
            # if cast target is fp16, change it to fp32
            for attr in node_pb.attribute:
                if attr.name == "to" and attr.i == TensorProto.FLOAT16:
                    attr.i = TensorProto.FLOAT
    for init in graph.initializer:
        if init.data_type == TensorProto.FLOAT16:
            array = to_array(init).astype("float32")
            init.data_type = TensorProto.FLOAT
            init.raw_data = from_array(array).raw_data
    for io in itertools.chain(graph.input, graph.output):
        if io.type.tensor_type.elem_type == TensorProto.FLOAT16:
            io.type.tensor_type.elem_type = TensorProto.FLOAT
    return graph


@PASSES.register()
def float_to_half(graph: OnnxGraph) -> OnnxGraph:
    """Convert float32 consts and values to half."""

    for node in graph:
        node_pb = graph.nodes[node]["pb"]
        if node_pb.op_type == "Constant":
            tensor = node_pb.attribute[0].t
            if tensor.data_type == TensorProto.FLOAT:
                array = to_array(tensor).astype("float16")
                attr = make_attribute(key="value", value=from_array(array))
                node_pb.attribute.pop()
                node_pb.attribute.append(attr)
    for init in graph.initializer:
        if init.data_type == TensorProto.FLOAT:
            array = to_array(init).astype("float16")
            init.data_type = TensorProto.FLOAT16
            init.raw_data = from_array(array).raw_data
    for io in itertools.chain(graph.input, graph.output):
        if io.type.tensor_type.elem_type == TensorProto.FLOAT:
            io.type.tensor_type.elem_type = TensorProto.FLOAT16
    return graph
