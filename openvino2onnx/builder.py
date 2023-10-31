"""
Copyright Wenyi Tang 2023

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

"""

import io
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


def build(g: nx.DiGraph, version: int = 13) -> ModelProto:  # noqa: C901
    """Build a graph to onnx model.

    The graph is a DiGraph object parsed from :func:`~openvino2onnx.ir11.ir_to_graph`.
    Some complex IR needs some transformations to a more constraint IR that we call it
    `legalization`. See :func:`~openvino2onnx.legalize.legalize`.

    Args:
        g (nx.DiGraph): the graph
        version (int): specify onnx import opset version

    Returns:
        onnx.ModelProto: onnx model
    """
    onnx_graph = GraphProto(
        name="openvino2onnx", doc_string=r"ONNX graph converted from OpenVino IR"
    )
    force_fp32 = g.graph.get("force_fp32")
    for i in itertools.chain(g.graph["input"], g.graph["output"]):
        attr = g.nodes[i]
        port = attr["outputs"]["0"] if i in g.graph["input"] else attr["inputs"]["0"]
        if precision := port.get("precision"):
            tensor_type = DTYPE2TENSORTYPE[np.dtype(PREC2DTYPE[precision])]
        else:
            tensor_type = None
        if tensor_type == DTYPE2TENSORTYPE[np.dtype("float16")] and force_fp32:
            tensor_type = DTYPE2TENSORTYPE[np.dtype("float32")]
        dims = [TensorShapeProto.Dimension(dim_value=int(i)) for i in port["dim"]]
        tensor = TypeProto.Tensor(elem_type=tensor_type)
        tensor.shape.dim.extend(dims)
        v = ValueInfoProto(name=attr["name"], type=TypeProto(tensor_type=tensor))
        if i in g.graph["input"]:
            onnx_graph.input.append(v)
        else:
            onnx_graph.output.append(v)

    for i in nx.topological_sort(g):
        if i in g.graph["input"] or i in g.graph["output"]:
            continue
        attr = g.nodes[i]
        try:
            if attr["type"] == "Const":
                if len(attr["outputs"]) != 1:
                    raise RuntimeError(
                        "Expect Const's output port number is 1, "
                        f"but there are {len(attr['outputs'])} ports"
                    )
                port = list(attr["outputs"].values())[0]
                dims = map(int, port["dim"])
                tensor = TensorProto(
                    name=attr["name"],
                    dims=list(dims),
                    data_type=DTYPE2TENSORTYPE[np.dtype(PREC2DTYPE[port["precision"]])],
                    raw_data=attr["data"].tobytes(),
                )
                onnx_graph.initializer.append(tensor)
                continue
            onnx_op = NodeProto(name=attr["name"])
            input_dict = {}
            for pred in g.predecessors(i):
                pred_attr = g.nodes[pred]
                if pred_attr["type"] in ("Parameter", "Const"):
                    in_port = g[pred][i][0]["dst"]
                    input_dict[in_port] = pred_attr["name"]
                else:
                    res, src = find_connect_to_output(g, pred)
                    for uv in g[pred][i].values():
                        in_port = uv["dst"]
                        if src == uv["src"]:
                            # port also connects to output
                            name = g.nodes[res]["name"]
                        else:
                            name = pred_attr["outputs"][uv["src"]]["name"]
                            name = f"{pred_attr['name']}/{name}"
                        input_dict[in_port] = name
            for pred in attr["inputs"]:
                if pred not in input_dict and attr["inputs"][pred].get("empty"):
                    input_dict[pred] = ""
            onnx_op.input.extend([input_dict[k] for k in sorted(input_dict)])
            # dangled output?
            res, src = find_connect_to_output(g, i)
            for i, out in attr["outputs"].items():
                if i == src:
                    onnx_op.output.append(g.nodes[res]["name"])
                else:
                    onnx_op.output.append(f"{attr['name']}/{out['name']}")
            op_type, attrs = get_onnx_optype_and_attributes(attr)
            onnx_op.op_type = op_type
            onnx_op.attribute.extend(attrs)
            onnx_graph.node.append(onnx_op)
        except Exception:
            errmsg = io.StringIO()
            errmsg.write(f"error on {attr['type']} node {attr['name']}\n")
            for k, v in attr.items():
                if k in ("type", "name", "data"):
                    continue
                errmsg.write(f"  {k}: {v}\n")
            print(errmsg.getvalue())
            raise

    model = ModelProto(
        ir_version=onnx.IR_VERSION,
        producer_name="openvino2onnx",
        graph=onnx_graph,
        opset_import=[OperatorSetIdProto(version=version)],
    )
    try:
        model = onnx.shape_inference.infer_shapes(
            model, check_type=True, strict_mode=True
        )
    except onnx.shape_inference.InferenceError:
        model = onnx.shape_inference.infer_shapes(model)
        with tempfile.NamedTemporaryFile("wb", suffix=".onnx", delete=False) as file:
            onnx.save(model, file.name)
        print("dump model to ", file.name)
        raise
    onnx.checker.check_model(model)
    return model
