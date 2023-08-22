"""
Copyright Wenyi Tang 2023

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

"""

from os import PathLike
from pathlib import Path
from typing import Optional, Union
from xml.etree import ElementTree

import networkx as nx
import numpy as np

from .mapping import ETYPE2DTYPE


def _node_to_dict(_node):
    return {i[0]: i[1] for i in _node.items()}


def _add_layer(graph, layer):
    node = layer.get("id")
    graph.add_node(
        node,
        name=layer.get("name"),
        type=layer.get("type"),
        version=layer.get("version"),
    )
    if (data := layer.find("data")) is not None:
        attrs = _node_to_dict(data)
        graph.add_node(node, **attrs)
    if (inputs := layer.find("input")) is not None:
        input_list = {}
        for port in inputs.iterfind("port"):
            port_id = port.get("id")
            input_list[port_id] = _node_to_dict(port)
            input_list[port_id].update(dim=[i.text for i in port.iterfind("dim")])
            if (attr := port.find("rt_info/attribute")) is not None:
                input_list[port_id].update(attr=_node_to_dict(attr))
        graph.add_node(node, inputs=input_list)
    if (outputs := layer.find("output")) is not None:
        output_list = {}
        for port in outputs.iterfind("port"):
            port_id = port.get("id")
            output_list[port_id] = _node_to_dict(port)
            output_list[port_id].update(dim=[i.text for i in port.iterfind("dim")])
            if (attr := port.find("rt_info/attribute")) is not None:
                output_list[port_id].update(attr=_node_to_dict(attr))
            if "name" not in output_list[port_id]:
                output_list[port_id].update(name=f"output{port_id}")
        graph.add_node(node, outputs=output_list)


def _add_edge(graph: nx.DiGraph, edge):
    beg = edge.get("from-layer")
    end = edge.get("to-layer")
    graph.add_edge(beg, end, src=edge.get("from-port"), dst=edge.get("to-port"))


def _load_const(graph, model_bin):
    for node in graph:
        try:
            if graph.nodes[node]["type"] != "Const":
                continue
            const_node = graph.nodes[node]
            size = int(const_node["size"])
            offset = int(const_node["offset"])
            dtype = ETYPE2DTYPE[const_node["element_type"]]
            if const_node["shape"]:
                shape = list(map(int, const_node["shape"].split(",")))
            else:
                shape = [-1]
            raw = np.fromfile(model_bin, dtype="uint8", count=size, offset=offset)
            data = np.frombuffer(raw.tobytes(), dtype=dtype).reshape(shape)
            const_node["data"] = data
        except Exception:
            print(f"exceptions on node:{node} {graph.nodes[node]['name']}")
            raise


def _set_input_and_output(graph):
    inputs = []
    outputs = []
    for node in graph:
        if graph.nodes[node]["type"] == "Parameter":
            inputs.append(node)
        elif graph.nodes[node]["type"] == "Result":
            outputs.append(node)
    graph.graph["input"] = inputs
    graph.graph["output"] = outputs


def ir_to_graph(
    model_path: Union[str, PathLike], model_bin: Optional[Union[str, PathLike]] = None
) -> Union[nx.MultiDiGraph, nx.DiGraph]:
    """Parse OpenVINO IR format XML to ``DiGraph`` or ``MultiDiGraph``.

    Args:
        model_path (Union[str, PathLike]): A URL to model xml file.
        model_bin (Optional[Union[str, PathLike]], optional): A URL to model bin file.
            If not specified, search at the same directory of xml file with same name.

    Raises:
        NotImplementedError: If IR version it not supported.

    Returns:
        Union[nx.MultiDiGraph, nx.DiGraph]: a graph
    """
    etree = ElementTree.parse(model_path)
    name = etree.getroot().get("name")
    if (ir_ver := etree.getroot().get("version")) not in ("10", "11"):
        raise NotImplementedError(
            f"IRv{ir_ver} is not supported, expected IRv10 or IRv11"
        )

    if model_bin is None and Path(model_path).with_suffix(".bin").exists():
        model_bin = Path(model_path).with_suffix(".bin")
    graph_info = {}
    if (rt_info := etree.find("rt_info")) is not None:
        graph_info = {
            i.tag: i.get("value")
            for i in filter(lambda x: "value" in x.keys(), rt_info.iter())
        }
    if "name" in graph_info:
        graph_info.pop("name")
    graph = nx.MultiDiGraph(name=name, **graph_info)
    for layer in etree.iterfind("layers/layer"):
        _add_layer(graph, layer)
    if model_bin is not None:
        _load_const(graph, model_bin)
    for edge in etree.iterfind("edges/edge"):
        _add_edge(graph, edge)
    _set_input_and_output(graph)
    return graph
