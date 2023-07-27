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
class Reshape(SingleNodeMutator):
    """Calculate reshape accurate shape"""

    def __init__(self):
        super().__init__(pattern="Reshape")

    def trans(self, graph: nx.MultiDiGraph, node):
        attrs = graph.nodes[node]
        # treat 0 as a wildcard to copy from the input on same axis
        special_zero = attrs["special_zero"]
        shape_is_const = False
        with suppress(Exception):
            shape = fold_const_on_node(graph, node, "1")
            shape_is_const = True
        if shape_is_const:
            ref_data = np.empty(list(map(int, attrs["inputs"]["0"]["dim"])))
            if special_zero:
                for i, d in enumerate(shape):
                    if d == 0:
                        shape[i] = ref_data.shape[i]
            shape = ref_data.reshape(shape).shape
            attrs["inputs"].pop("1")
            expand_const_on_node(graph, node, np.array(shape, "int64"), "1")


@legalize.register
class Unsqueeze(SingleNodeMutator):
    """legalize axes shape"""

    def __init__(self):
        super().__init__(pattern="Unsqueeze")

    def trans(self, graph: nx.MultiDiGraph, node):
        attrs = graph.nodes[node]
        axes = fold_const_on_node(graph, node, "1")
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
        const = fold_const_on_node(graph, node, "1")
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
        const = fold_const_on_node(graph, node, "2")
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
        steps = fold_const_on_node(graph, node, "3")
        assert steps.ndim == 1
        attrs["inputs"]["3"].update(empty=True)
        expand_const_on_node(graph, node, steps, "4")
        begin_mask = list(map(int, attrs["begin_mask"].split(",")))
        end_mask = list(map(int, attrs["end_mask"].split(",")))
        if any(i != 0 for i in begin_mask + end_mask):
            # adjust begin and end
            data_shape = list(map(int, attrs["inputs"]["0"]["dim"]))
            begin_var = fold_const_on_node(graph, node, "1")
            end_var = fold_const_on_node(graph, node, "2")
            for i, x in enumerate(begin_mask):
                begin_var[i] = 0 if x != 0 else begin_var[i]
            for i, x in enumerate(end_mask):
                end_var[i] = data_shape[i] if x != 0 else end_var[i]
            attrs["inputs"].pop("1")
            attrs["inputs"].pop("2")
            expand_const_on_node(graph, node, begin_var, "1")
            expand_const_on_node(graph, node, end_var, "2")
        if not attrs["shrink_axis_mask"]:
            shrink_axis_mask = []
        else:
            shrink_axis_mask = list(map(int, attrs["shrink_axis_mask"].split(",")))
        if any(i != 0 for i in shrink_axis_mask):
            # add a squeeze
            squeeze_node = f"{node}_squeeze"
            outport = list(attrs["outputs"].keys())[0]
            graph.add_node(
                squeeze_node,
                name=squeeze_node,
                type="Squeeze",
                version="opset1",
                inputs={"0": attrs["outputs"][outport]},
                outputs={"2": dict(name=f"{squeeze_node}/output2")},
            )
            for u, v, data in list(graph.out_edges(node, True)):
                graph.add_edge(squeeze_node, v, src="2", dst=data["dst"])
                graph.remove_edge(u, v)
            graph.add_edge(node, squeeze_node, src=outport, dst="0")
            (axis,) = np.nonzero(shrink_axis_mask)
            expand_const_on_node(graph, squeeze_node, axis, "1")
            attrs["shrink_axis_mask"] = ""


@legalize.register
class Interpolate(SingleNodeMutator):
    """To Resize"""

    def __init__(self):
        super().__init__(pattern="Interpolate")

    def trans(self, graph: nx.MultiDiGraph, node):
        attrs = graph.nodes[node]
        # get scales_or_size
        scales_or_size = fold_const_on_node(graph, node, "1")
        # size to scales
        input_dims = np.array(list(map(int, attrs["inputs"]["0"]["dim"])))
        scales = scales_or_size
        with suppress(ValueError):
            np.iinfo(scales_or_size.dtype)
            if scales.size < input_dims.size:
                scales = np.concatenate(
                    [input_dims[: input_dims.size - scales.size], scales]
                )
            scales = np.true_divide(scales, input_dims).astype("float32")
        if scales.size < input_dims.size:
            scales = np.concatenate(
                [input_dims[: input_dims.size - scales.size], scales]
            )
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
        axis_var = fold_const_on_node(graph, node, "1")
        attrs["inputs"].pop("1")
        expand_const_on_node(graph, node, axis_var.astype("int64"), "1")
