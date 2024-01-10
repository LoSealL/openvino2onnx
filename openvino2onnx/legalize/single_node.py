"""
Copyright Wenyi Tang 2023

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

Split IR op to more than 1 onnx op
"""
# pylint: disable=missing-function-docstring, missing-class-docstring

import copy
import warnings
from contextlib import suppress
from itertools import product

import networkx as nx
import numpy as np
import onnx

from openvino2onnx.mapping import PREC2DTYPE

from .compose import legalize
from .fold_const import expand_const_on_node, fold_const_on_node, make_output_for_node
from .mutator import SingleNodeMutator
from .utils import get_node_on_edge, text_to_boolean


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
                ndim = len(output["dim"])
                perm = list(range(ndim))
                perm[-2:] = ndim - 1, ndim - 2
                graph.add_node(
                    transpose,
                    name=f"{attrs['name']}_transpose_{port}",
                    type="Transpose",
                    version="opset1",
                    inputs={"0": attrs["inputs"][order]},
                    outputs={"1": output},
                    perm=",".join(map(str, perm)),
                )
                graph.add_edge(u, transpose, src=data["src"], dst=order)
                graph.add_edge(transpose, node, src="1", dst=data["dst"])


@legalize.register
class GroupConvolution(SingleNodeMutator):
    """Reshape weights to 4D and extract group attribute"""

    def __init__(self):
        super().__init__(pattern="GroupConvolution")

    def trans(self, graph: nx.MultiDiGraph, node):
        attrs = graph.nodes[node]
        weight = fold_const_on_node(graph, node, "1")
        groups = weight.shape[0]
        weight = weight.reshape([-1, *weight.shape[2:]])
        attrs["inputs"].pop("1")
        expand_const_on_node(graph, node, weight, "1")
        attrs["group"] = groups


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
        if "2" in attrs["inputs"]:
            attrs["inputs"].pop("2")
        expand_const_on_node(graph, node, scales, "2")
        # axes
        if "3" in attrs["inputs"]:
            axes = fold_const_on_node(graph, node, "3")
            attrs["inputs"].pop("3")
            assert tuple(axes) == (2, 3)


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


@legalize.register
class Swish(SingleNodeMutator):
    """Change swish to mul and sigmoid.

    Swish(x) = x * sigmoid(x)
    """

    def __init__(self):
        super().__init__(pattern="Swish")

    def trans(self, graph: nx.MultiDiGraph, node):
        attrs = graph.nodes[node]
        preds = {k: graph[k][node] for k in graph.predecessors(node)}
        succs = {k: graph[node][k] for k in graph.successors(node)}
        # remove swish
        graph.remove_node(node)
        # add sigmoid
        sigmoid_node = f"{node}_sigmoid"
        graph.add_node(
            sigmoid_node,
            name=sigmoid_node,
            type="Sigmoid",
            version="opset1",
            inputs=copy.deepcopy(attrs["inputs"]),
            outputs=copy.deepcopy(attrs["outputs"]),
        )
        # add mul
        mul_node = f"{node}_mul"
        graph.add_node(
            mul_node,
            name=mul_node,
            type="Multiply",
            version="opset1",
            inputs=copy.deepcopy(attrs["inputs"]),
            outputs=copy.deepcopy(attrs["outputs"]),
        )
        mul_attrs = graph.nodes[mul_node]
        mul_attrs["inputs"]["1"] = mul_attrs["inputs"]["0"]
        mul_attrs["inputs"]["1"].update(id=1)
        mul_attrs["outputs"]["2"] = mul_attrs["outputs"].pop("1")
        mul_attrs["outputs"]["2"].update(id=2, name="output2")
        # add edges back
        graph.add_edge(sigmoid_node, mul_node, src="1", dst="1")
        for i in preds:
            graph.add_edge(i, sigmoid_node, src=preds[i][0]["src"], dst="0")
            graph.add_edge(i, mul_node, src=preds[i][0]["src"], dst="0")
        for i in succs:
            graph.add_edge(mul_node, i, src="2", dst=succs[i][0]["dst"])


@legalize.register
class Pad(SingleNodeMutator):
    """Combine inputs 1 & 2 to pads"""

    def __init__(self):
        super().__init__(pattern="Pad")

    def trans(self, graph: nx.MultiDiGraph, node):
        attrs = graph.nodes[node]
        const_value = fold_const_on_node(graph, node, "3")
        begin = fold_const_on_node(graph, node, "1")
        end = fold_const_on_node(graph, node, "2")
        attrs["inputs"].pop("1")
        attrs["inputs"].pop("2")
        attrs["inputs"].pop("3")
        expand_const_on_node(graph, node, np.concatenate([begin, end]), "1")
        expand_const_on_node(graph, node, const_value, "2")


class ReduceOp(SingleNodeMutator):
    """Until Opset 13 ReduceOp has only 1 inputs"""

    def __init__(self):
        super().__init__(pattern=self.__class__.__name__)

    def trans(self, graph: nx.MultiDiGraph, node):
        attrs = graph.nodes[node]
        if "1" in attrs["inputs"]:
            attrs["axes"] = fold_const_on_node(graph, node, "1")


@legalize.register
class ReduceMean(ReduceOp):
    ...


@legalize.register
class ReduceMax(ReduceOp):
    ...


@legalize.register
class ReduceMin(ReduceOp):
    ...


@legalize.register
class ReduceProd(ReduceOp):
    ...


@legalize.register
class ReduceSum(ReduceOp):
    ...


@legalize.register
class ReduceSumSquare(ReduceOp):
    ...


@legalize.register
class VariadicSplit(SingleNodeMutator):
    """move input axis to attribute"""

    def __init__(self):
        super().__init__(pattern="VariadicSplit")

    def trans(self, graph: nx.MultiDiGraph, node):
        attrs = graph.nodes[node]
        if "1" in attrs["inputs"]:
            attrs["axis"] = fold_const_on_node(graph, node, "1").flatten()
            attrs["inputs"].pop("1")


@legalize.register
class PReLU(SingleNodeMutator):
    """Legalize parameters data type for PRelu"""

    def __init__(self):
        super().__init__(pattern="PReLU")

    def trans(self, graph: nx.MultiDiGraph, node):
        attrs = graph.nodes[node]
        if prec := attrs["outputs"]["2"].get("precision"):
            prec = PREC2DTYPE[prec]
            parameter = fold_const_on_node(graph, node, "1").astype(prec)
            attrs["inputs"].pop("1")
            expand_const_on_node(graph, node, parameter, "1")


@legalize.register
class Clamp(SingleNodeMutator):
    """Transmit attribute of clamp node to input ports"""

    def __init__(self):
        super().__init__(pattern="Clamp")

    def trans(self, graph: nx.MultiDiGraph, node):
        attrs = graph.nodes[node]
        min_value = float(attrs["min"])
        max_value = float(attrs["max"])
        if prec := attrs["outputs"]["1"].get("precision"):
            dtype = PREC2DTYPE[prec]
        elif prec := attrs["inputs"]["0"].get("precision"):
            dtype = PREC2DTYPE[prec]
        else:
            raise ValueError("Can't deduce min/max data type of clip node")
        min_value = np.array(min_value, dtype=dtype)
        max_value = np.array(max_value, dtype=dtype)
        expand_const_on_node(graph, node, min_value)
        expand_const_on_node(graph, node, max_value)


@legalize.register
class FakeQuantizeQDQ(SingleNodeMutator):
    """Transmit fake quantize to QDQ nodes.

    There's a bug in onnx that DequanzeLinear doesn't support fp16 well:
    https://github.com/onnx/onnx/issues/5704
    """

    def __init__(self):
        super().__init__(pattern="FakeQuantize")

    def trans(self, graph: nx.MultiDiGraph, node):
        attrs = graph.nodes[node]
        levels = int(attrs.get("levels", 256))
        nbits = int(np.log2(levels))
        assert nbits in (8, 16, 32)
        input_low = fold_const_on_node(graph, node, "1").squeeze()
        input_high = fold_const_on_node(graph, node, "2").squeeze()
        output_low = fold_const_on_node(graph, node, "3").squeeze()
        output_high = fold_const_on_node(graph, node, "4").squeeze()
        for inp in ("1", "2", "3", "4"):
            attrs["inputs"].pop(inp)
        # keep old edges
        preds = {k: graph[k][node] for k in graph.predecessors(node)}
        succs = {k: graph[node][k] for k in graph.successors(node)}
        # remove fake quantize
        graph.remove_node(node)
        # add QuantizeLinear
        q_node = f"{node}_q"
        graph.add_node(
            q_node,
            name=q_node,
            type="QuantizeLinear",
            version="opset1",
            inputs=copy.deepcopy(attrs["inputs"]),
            outputs=copy.deepcopy(attrs["outputs"]),
        )
        # add DequantizeLinear
        dq_node = f"{node}_dq"
        graph.add_node(
            dq_node,
            name=dq_node,
            type="DequantizeLinear",
            version="opset1",
            inputs=copy.deepcopy(attrs["inputs"]),
            outputs=copy.deepcopy(attrs["outputs"]),
        )
        scale_prec = PREC2DTYPE[attrs["outputs"]["5"].get("precision", "FP32")]
        zero_prec = np.iinfo(f"uint{nbits}").dtype
        scales = ((input_high - input_low) / (levels - 1)).astype(scale_prec)
        zero_points = np.rint((levels - 1) * input_low / (input_low - input_high))
        zero_points = zero_points.astype(zero_prec)
        expand_const_on_node(graph, q_node, scales, "1")
        expand_const_on_node(graph, q_node, zero_points, "2")
        scales = ((output_high - output_low) / (levels - 1)).astype(scale_prec)
        zero_points = np.rint((levels - 1) * output_low / (output_low - output_high))
        zero_points = zero_points.astype(zero_prec)
        expand_const_on_node(graph, dq_node, scales, "1")
        expand_const_on_node(graph, dq_node, zero_points, "2")
        graph.add_edge(q_node, dq_node, src="5", dst="0")
        # restore old edges
        for i in preds:
            graph.add_edge(i, q_node, src=preds[i][0]["src"], dst="0")
        for i in succs:
            graph.add_edge(dq_node, i, src="5", dst=succs[i][0]["dst"])


@legalize.register
class PriorBoxClustered(SingleNodeMutator):
    """Fold PriorBoxClustered node to a constant."""

    def __init__(self):
        super().__init__(pattern="PriorBoxClustered")

    def trans(self, graph: nx.MultiDiGraph, node):
        # widths         Desired widths of prior boxes
        # heights        Desired heights of prior boxes
        # clip           Clip output to [0,1]
        # step_widths    Distance between prior box centers
        # step_heights   Distance between prior box centers
        # step           Distance between prior box centers (when step_w = step_h)
        # offset         Box offset relative to top center of image
        # variances      Values to adjust prior boxes with
        attrs = graph.nodes[node]
        widths = list(map(float, attrs["width"].split(",")))
        heights = list(map(float, attrs["height"].split(",")))
        clip = text_to_boolean(attrs.get("clip", "1"))
        step_widths = float(attrs.get("step_w", "0"))
        step_heights = float(attrs.get("step_h", "0"))
        step = float(attrs.get("step", "0"))
        offset = float(attrs.get("offset", "0"))
        variances = list(map(float, attrs.get("variance", "").split(",")))
        if len(widths) != len(heights):
            raise ValueError(
                f"Size of heights vector: {heights}"
                f" doesn't match size of widths vector: {widths}"
            )
        if len(variances) == 0:
            variances.append(0.1)
        if len(variances) not in (1, 4):
            raise ValueError(f"variance size must be 0, 1 or 4, got {variances}")
        if step_widths == 0:
            step_widths = step
        if step_heights == 0:
            step_heights = step

        prec = attrs["outputs"]["2"].get("precision", "FP32")
        anchers = self._fold_anchers(
            graph,
            node,
            widths,
            heights,
            step_widths,
            step_heights,
            variances,
            offset,
            clip,
        ).astype(PREC2DTYPE[prec])
        attrs.clear()
        graph.add_node(
            node,
            name=node,
            type="Const",
            version="opset1",
            shape=",".join(map(str, anchers.shape)),
            outputs={"0": dict(precision=prec, dim=list(map(str, anchers.shape)))},
            data=anchers,
        )
        # clear predecessors
        graph.remove_nodes_from(graph.predecessors(node))

    def _fold_anchers(
        self, graph, node, width, height, step_w, step_h, variance, offset, clip
    ):
        # feature map size
        data_size = fold_const_on_node(graph, node, "0")
        # image size
        img_size = fold_const_on_node(graph, node, "1")
        layer_h, layer_w = data_size
        img_h, img_w = img_size
        if step_w == 0 and step_h == 0:
            step_w = img_w / layer_w
            step_h = img_h / layer_h
        x, y = np.meshgrid(range(layer_h), range(layer_w))
        grid = np.stack([x, y], axis=-1)
        center = (grid + offset) * (step_w, step_h)
        # [h, w, priors, 2]
        centers = np.tile(center[..., None], len(width)).transpose([0, 1, 3, 2])
        box_size = np.stack([width, height], -1)
        x0y0 = (centers - box_size / 2) / (img_w, img_h)
        x1y1 = (centers + box_size / 2) / (img_w, img_h)
        if clip:
            x0y0 = np.clip(x0y0, 0, 1)
            x1y1 = np.clip(x1y1, 0, 1)
        anchers = np.concatenate([x0y0, x1y1], axis=-1).flatten()
        if len(variance) == 1:
            variance = np.tile(variance, 4)
        variance = np.tile(variance, anchers.size // 4).flatten()
        return np.stack([anchers, variance])
