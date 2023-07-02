"""
Copyright Wenyi Tang 2023

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

Split IR op to more than 1 onnx op
"""

import copy
from contextlib import suppress
from itertools import product

import networkx as nx
import numpy as np

from openvino2onnx.mapping import PREC2DTYPE

from .compose import legalize
from .fold_const import expand_const_on_node, fold_const_on_node
from .mutator import SingleNodeMutator
from .utils import get_node_on_edge


@legalize.register
class ShapeOf(SingleNodeMutator):
    """Change ShapeOf to Const"""

    def __init__(self):
        super().__init__(pattern="ShapeOf")

    def trans(self, graph: nx.MultiDiGraph, node):
        attrs = graph.nodes[node]
        graph.remove_edges_from(list(graph.in_edges(node)))
        dims = [i["dim"] for i in attrs["inputs"].values()]
        assert len(dims) == 1
        dim = list(map(int, dims[0]))
        output = attrs["outputs"].pop("1")
        attrs["type"] = "Const"
        attrs["data"] = np.array(dim, dtype=PREC2DTYPE[output["precision"]])
        attrs["inputs"].clear()
        attrs["outputs"]["0"] = output


@legalize.register
class Unsqueeze(SingleNodeMutator):
    """legalize axes shape"""

    def __init__(self):
        super().__init__(pattern="Unsqueeze")

    def trans(self, graph: nx.MultiDiGraph, node):
        attrs = graph.nodes[node]
        u = get_node_on_edge(graph, node, "1")
        axes = fold_const_on_node(graph, u)
        assert axes.ndim == 1
        attrs["axes"] = axes.flatten()
        attrs["inputs"].pop("1")
        expand_const_on_node(graph, node, np.array(attrs["axes"], "int64"), "1")


@legalize.register
class MatMul(SingleNodeMutator):
    """Split matmul to transpose + matmul"""

    def __init__(self):
        super().__init__(pattern="MatMul")

    def trans(self, graph: nx.MultiDiGraph, node):
        attrs = graph.nodes[node]
        ports = ("a", "0"), ("b", "1")
        for (u, v, data), (port, order) in product(graph.in_edges(node, True), ports):
            if attrs[f"transpose_{port}"] == "true" and data["dst"] == order:
                graph.remove_edge(u, v)
                output = copy.deepcopy(attrs["inputs"][order])
                output["dim"] = output["dim"][::-1]
                output["name"] = "output1"
                transpose = f"{node}_transpose_{port}"
                graph.add_node(
                    transpose,
                    name=f"{attrs['name']}_transpose_{port}",
                    type="Transpose",
                    version="opset1",
                    inputs={"0": attrs["inputs"][order]},
                    outputs={"1": output},
                    perm="1, 0",
                )
                graph.add_edge(u, transpose, src=data["src"], dst=order)
                graph.add_edge(transpose, node, src="1", dst=data["dst"])


@legalize.register
class Transpose(SingleNodeMutator):
    """Fold const input:1 as attribute perm."""

    def __init__(self):
        super().__init__(pattern="Transpose")

    def trans(self, graph: nx.MultiDiGraph, node):
        attrs = graph.nodes[node]
        if "perm" in attrs:
            return
        # find a solo path to input:1
        u = get_node_on_edge(graph, node, "1")
        const = fold_const_on_node(graph, u)
        assert const.ndim == 1
        attrs["inputs"].pop("1")
        attrs.update(perm=",".join(map(str, const.tolist())))


@legalize.register
class Gather(SingleNodeMutator):
    """Evaluate axis to attr"""

    def __init__(self):
        super().__init__(pattern="Gather")

    def trans(self, graph: nx.MultiDiGraph, node):
        attrs = graph.nodes[node]
        u = get_node_on_edge(graph, node, "2")
        const = fold_const_on_node(graph, u)
        assert const.size == 1
        attrs["axis"] = int(const.flatten()[0])
        attrs["inputs"].pop("2")


@legalize.register
class StridedSlice(SingleNodeMutator):
    """Swap axes and steps"""

    def __init__(self):
        super().__init__(pattern="StridedSlice")

    def trans(self, graph: nx.MultiDiGraph, node):
        attrs = graph.nodes[node]
        strides = get_node_on_edge(graph, node, "3")
        steps = fold_const_on_node(graph, strides)
        assert steps.ndim == 1
        attrs["inputs"]["3"].update(empty=True)
        expand_const_on_node(graph, node, steps, "4")
        begin_mask = list(map(int, attrs["begin_mask"].split(",")))
        end_mask = list(map(int, attrs["end_mask"].split(",")))
        if any(i != 0 for i in begin_mask + end_mask):
            # adjust begin and end
            begin = get_node_on_edge(graph, node, "1")
            end = get_node_on_edge(graph, node, "2")
            data_shape = list(map(int, attrs["inputs"]["0"]["dim"]))
            begin_var = fold_const_on_node(graph, begin)
            end_var = fold_const_on_node(graph, end)
            for i, x in enumerate(begin_mask):
                begin_var[i] = 0 if x != 0 else begin_var[i]
            for i, x in enumerate(end_mask):
                end_var[i] = data_shape[i] if x != 0 else end_var[i]
            attrs["inputs"].pop("1")
            attrs["inputs"].pop("2")
            expand_const_on_node(graph, node, begin_var, "1")
            expand_const_on_node(graph, node, end_var, "2")


@legalize.register
class Interpolate(SingleNodeMutator):
    """To Resize"""

    def __init__(self):
        super().__init__(pattern="Interpolate")

    def trans(self, graph: nx.MultiDiGraph, node):
        attrs = graph.nodes[node]
        # get scales_or_size
        scales_or_size = get_node_on_edge(graph, node, "1")
        scales_or_size = fold_const_on_node(graph, scales_or_size)
        # size to scales
        input_dims = list(map(int, attrs["inputs"]["0"]["dim"]))
        scales = scales_or_size
        with suppress(ValueError):
            np.iinfo(scales_or_size.dtype)
            scales = np.true_divide(scales_or_size, input_dims).astype("float32")
        # make empty roi
        attrs["inputs"]["1"].update(empty=True)
        # make scales
        expand_const_on_node(graph, node, scales, "2")


@legalize.register
class Squeeze(SingleNodeMutator):
    """Cast axis from U64 to I64"""

    def __init__(self):
        super().__init__(pattern="Squeeze")

    def trans(self, graph: nx.MultiDiGraph, node):
        attrs = graph.nodes[node]
        axis = get_node_on_edge(graph, node, "1")
        axis_var = fold_const_on_node(graph, axis)
        attrs["inputs"].pop("1")
        expand_const_on_node(graph, node, axis_var.astype("int64"), "1")
