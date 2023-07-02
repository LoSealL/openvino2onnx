"""
Copyright Wenyi Tang 2023

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

"""

import networkx as nx
import numpy as np
from onnx.reference import ReferenceEvaluator

from openvino2onnx.builder import build
from openvino2onnx.mapping import DTYPE2PREC, PREC2DTYPE


def _make_output_for_node(graph: nx.DiGraph, node, port=None):
    attrs = graph.nodes[node]
    if port is None:
        port = list(attrs["outputs"].keys())[0]
    result_node = f"{node}_result{port}"
    input_port = attrs["outputs"][port]
    input_port.update(id="0", name=result_node)
    graph.add_node(
        result_node,
        name=result_node,
        type="Result",
        version="opset1",
        inputs={"0": input_port},
    )
    graph.add_edge(node, result_node, src=port, dst="0")
    return result_node


def fold_const_on_node(graph: nx.DiGraph, node, remove_nodes=True) -> np.ndarray:
    """Fold the const path to node in the graph.

    Args:
        graph (nx.DiGraph): the graph
        node (str): a node in the graph that can be folded
        remove_nodes (bool, optional): If true, remove the folded node from graph.
            Defaults to True.

    Returns:
        ndarray: evaluated const data
    """
    attrs = graph.nodes[node]
    sources = nx.ancestors(graph, node)
    subg: nx.DiGraph = nx.subgraph(graph, list(sources) + [node]).copy()
    # all sources in subgraph must be Const
    for i in nx.topological_sort(subg):
        if subg.in_degree(i) != 0:
            break
        if subg.nodes[i]["type"] != "Const":
            raise RuntimeError(f"node {attrs['name']} is not constant!")
    if len(subg) == 1:
        # quick solution for only 1 Const
        const = graph.nodes[next(iter(subg))]["data"]
    else:
        subg.graph["input"] = []
        subg.graph["output"] = [_make_output_for_node(subg, node)]
        folder = ReferenceEvaluator(build(subg))
        const = folder.run(None, {})[0]
    if remove_nodes:
        graph.remove_nodes_from(subg)
    for out in attrs["outputs"].values():
        if prec := out.get("precision"):
            const = np.array(const, dtype=PREC2DTYPE[prec])
    return const


def expand_const_on_node(graph: nx.DiGraph, node, data, port=None):
    """Add a const node of data to the node input.

    Args:
        graph (nx.DiGraph): the graph
        node (str):  a node in graph to expand
        data (ndarray): const data
        port (str, optional): specify a input port id. Defaults to None.
    """
    attrs = graph.nodes[node]
    if port is None:
        port = len(attrs["inputs"])
    if port in attrs["inputs"]:
        raise ValueError(f"input:{port} at {attrs['name']} is in use.")
    # make a Const
    const_node = f"{node}_const{port}"
    graph.add_node(
        const_node,
        name=const_node,
        type="Const",
        version="opset1",
        shape=",".join(map(str, data.shape)),
        outputs={
            "0": dict(
                precision=DTYPE2PREC[np.dtype(data.dtype).name],
                dim=",".join(map(str, data.shape)),
            )
        },
        data=data,
    )
    graph.add_edge(const_node, node, src="0", dst=port)
