"""
Copyright Wenyi Tang 2023

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

"""

import itertools
import tempfile
from typing import Optional, Tuple

import networkx as nx
import numpy as np
import onnx
from onnx import (
    GraphProto,
    ModelProto,
    NodeProto,
    OperatorSetIdProto,
    TensorProto,
    TensorShapeProto,
    TypeProto,
    ValueInfoProto,
)

from openvino2onnx.ops import get_onnx_optype_and_attributes

from .mapping import DTYPE2TENSORTYPE, PREC2DTYPE


def find_connect_to_output(graph: nx.DiGraph, node: str) -> Tuple[Optional[str], str]:
    """Find if `node` has an edge connected to a graph output.

    Args:
        graph (DiGraph): the graph
        node (str): the node index

    Returns:
        Tuple[Optional[str], str]: a tuple of result node and output port id.
    """
    for succ in graph.successors(node):
        if graph.nodes[succ]["type"] == "Result":
            return succ, graph[node][succ][0]["src"]
    return None, ""


def build(g: nx.DiGraph) -> ModelProto:
    """Build a graph to onnx model.

    The graph is a DiGraph object parsed from :func:`~openvino2onnx.ir11.ir_to_graph`.
    Some complex IR needs some transformations to a more constraint IR that we call it
    `legalization`. See :func:`~openvino2onnx.legalize.legalize`.

    Args:
        g (nx.DiGraph): the graph

    Returns:
        onnx.ModelProto: onnx model
    """
    onnx_graph = GraphProto(
        name="openvino2onnx", doc_string=r"ONNX graph converted from OpenVino IR"
    )

    for i in itertools.chain(g.graph["input"], g.graph["output"]):
        node = g.nodes[i]
        port = node["outputs"]["0"] if i in g.graph["input"] else node["inputs"]["0"]
        tensor_type = DTYPE2TENSORTYPE[np.dtype(PREC2DTYPE[port["precision"]])]
        dims = [TensorShapeProto.Dimension(dim_value=int(i)) for i in port["dim"]]
        tensor = TypeProto.Tensor(elem_type=tensor_type)
        tensor.shape.dim.extend(dims)
        v = ValueInfoProto(name=node["name"], type=TypeProto(tensor_type=tensor))
        if i in g.graph["input"]:
            onnx_graph.input.append(v)
        else:
            onnx_graph.output.append(v)

    for i in nx.topological_sort(g):
        if i in g.graph["input"] or i in g.graph["output"]:
            continue
        node = g.nodes[i]
        if node["type"] == "Const":
            port = node["outputs"]["0"]
            dims = map(int, port["dim"])
            tensor = TensorProto(
                name=node["name"],
                dims=list(dims),
                data_type=DTYPE2TENSORTYPE[np.dtype(PREC2DTYPE[port["precision"]])],
                raw_data=node["data"].tobytes(),
            )
            onnx_graph.initializer.append(tensor)
            continue
        onnx_op = NodeProto(name=node["name"])
        input_dict = {}
        for pred in g.predecessors(i):
            pred_node = g.nodes[pred]
            dst = g[pred][i][0]["dst"]
            if pred_node["type"] in ("Parameter", "Const"):
                input_dict[dst] = pred_node["name"]
            else:
                res, src = find_connect_to_output(g, pred)
                for uv in g[pred][i].values():
                    if src == uv["src"]:
                        # port also connects to output
                        name = g.nodes[res]["name"]
                    else:
                        name = pred_node["outputs"][uv["src"]]["name"]
                        name = f"{pred_node['name']}/{name}"
                    input_dict[dst] = name
        for pred in node["inputs"]:
            if pred not in input_dict and node["inputs"][pred].get("empty"):
                input_dict[pred] = ""
        onnx_op.input.extend([input_dict[k] for k in sorted(input_dict)])
        # dangled output?
        res, src = find_connect_to_output(g, i)
        for i, out in node["outputs"].items():
            if i == src:
                onnx_op.output.append(g.nodes[res]["name"])
            else:
                onnx_op.output.append(f"{node['name']}/{out['name']}")
        op_type, attrs = get_onnx_optype_and_attributes(node)
        onnx_op.op_type = op_type
        onnx_op.attribute.extend(attrs)
        onnx_graph.node.append(onnx_op)

    onnx.checker.check_graph(onnx_graph)
    model = ModelProto(
        ir_version=onnx.IR_VERSION_2020_5_8,
        producer_name="openvino2onnx",
        graph=onnx_graph,
        opset_import=[OperatorSetIdProto(version=13)],
    )
    try:
        model = onnx.shape_inference.infer_shapes(
            model, check_type=True, strict_mode=True
        )
    except onnx.shape_inference.InferenceError:
        with tempfile.NamedTemporaryFile("wb", suffix=".onnx", delete=False) as file:
            onnx.save(model, file.name)
        print("dump model to ", file.name)
        raise
    return model
