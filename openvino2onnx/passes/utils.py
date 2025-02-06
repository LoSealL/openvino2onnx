"""
Copyright Wenyi Tang 2024-2025

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
from onnx import AttributeProto, NodeProto, TensorProto, numpy_helper, save_model
from onnx.helper import make_node

from .logger import debug


def make_constant(name: str, value: np.ndarray) -> "NodeProto":
    """Make a Constant node according to given value."""
    node = make_node(
        op_type="Constant",
        name=name,
        inputs=[],
        outputs=[f"{name}_output_0"],
        value=numpy_helper.from_array(value),
    )
    return node


def attribute_value(attr: AttributeProto):
    """Get the value of an onnx attribute."""
    match attr.type:
        case AttributeProto.FLOAT:
            return float(attr.f)
        case AttributeProto.INT:
            return int(attr.i)
        case AttributeProto.STRING:
            return attr.s.decode("utf-8")
        case AttributeProto.TENSOR:
            return numpy_helper.to_array(attr.t)
        case AttributeProto.TYPE_PROTO:
            return attr.tp
        case AttributeProto.FLOATS:
            return [float(f) for f in attr.floats]
        case AttributeProto.INTS:
            return [int(i) for i in attr.ints]
        case AttributeProto.STRINGS:
            return [s.decode("utf-8") for s in attr.strings]
        case AttributeProto.TENSORS:
            return [numpy_helper.to_array(t) for t in attr.tensors]
        case AttributeProto.GRAPHS:
            return [g for g in attr.graphs]
        case AttributeProto.TYPE_PROTOS:
            return [tp for tp in attr.type_protos]
    raise ValueError(f"Unsupported attribute type {attr.type}")


def canonical_node_name(name: str) -> str:
    """Canonicalize a node name by replacing illegal characters with '_'."""

    return name.replace(".", "_").replace(":", "_").replace("/", "_")


def is_elewise(node: NodeProto | str) -> bool:
    """Return True if the node type is an element-wise operation."""

    if isinstance(node, NodeProto):
        node = node.op_type
    return node in {
        "Abs",
        "Acos",
        "Acosh",
        "Add",
        "And",
        "Asin",
        "Asinh",
        "Atan",
        "Atanh",
        "BatchNormalization",
        "Bernoulli",
        "BitShift",
        "BitwiseAnd",
        "BitwiseNot",
        "BitwiseOr",
        "BitwiseXor",
        "Cast",
        "CastLike",
        "Ceil",
        "Celu",
        "Clip",
        "Cos",
        "Cosh",
        "CumSum",
        "Div",
        "Einsum",
        "Elu",
        "Equal",
        "Erf",
        "Exp",
        "Floor",
        "Gelu",
        "Greater",
        "GreaterOrEqual",
        "HardSigmoid",
        "HardSwish",
        "Hardmax",
        "Identity",
        "LeakyRelu",
        "Less",
        "LessOrEqual",
        "Log",
        "LogSoftmax",
        "LpNormalization",
        "Max",
        "Mean",
        "Min",
        "Mish",
        "Mod",
        "Mul",
        "Neg",
        "Not",
        "Or",
        "PRelu",
        "Pow",
        "Reciprocal",
        "Relu",
        "Round",
        "Selu",
        "Sigmoid",
        "Sign",
        "Sin",
        "Sinh",
        "Softmax",
        "Softplus",
        "Softsign",
        "Sqrt",
        "Sub",
        "Sum",
        "Tan",
        "Tanh",
        "Xor",
    }


def evaluate_on_node(
    graph, node: NodeProto, output_name: Optional[str] = None
) -> Optional[np.ndarray]:
    """Evaluate an output of a given node in the graph.

    Args:
        graph (OnnxGraph): the model graph.
        node (NodeProto): specify a node to evaluate its output.
        output_name (Optional[str], optional): specify the output name of the node.
            Defaults to None to use the first output of the node.

    Returns:
        Optional[np.ndarray]: a constant array value if succeed, None otherwise.
    """
    all_preds = nx.traversal.dfs_predecessors(graph.reverse(), node.name)
    all_preds = set(all_preds)
    all_preds.add(node.name)  # include self
    h = deepcopy(graph.onnx_subgraph(all_preds))
    try:
        # pylint: disable=import-outside-toplevel
        from openvino2onnx import convert_graph

        sub_model = h.model
        # extend current value_info due to ``h.model`` cleared it.
        # it's needed because shape inference can not work for custom domains.
        # pylint: disable=protected-access
        sub_model.graph.value_info.extend(h._value_info)
        sub_model = convert_graph(
            sub_model,
            [],
            print_passes=False,
            target_opset=graph.opset_version,
        ).model
    except Exception:  # pylint: disable=broad-except
        debug(f"convert subgraph failed on {node.name}: [{all_preds}]")
        return
    try:
        # pylint: disable=import-outside-toplevel
        from openvino2onnx.evaluator import Evaluator

        if output_name is None:
            output_name = sub_model.graph.output[0].name
        runner = Evaluator(sub_model)
        return runner([output_name], {})[0]
    except Exception:  # pylint: disable=broad-except
        node_name = canonical_node_name(node.name)
        temp_save = Path(tempfile.gettempdir()) / f"{node_name}.onnx"
        save_model(sub_model, temp_save)
        debug(f"evaluate subgraph failed on {node.name}: {temp_save}")
        return


def cast_in(node: NodeProto, index: int, to: int) -> NodeProto:
    """Insert a Cast node before the index-th input of the node.

    Args:
        node (NodeProto): specify a node to insert a Cast node.
        index (int): the index of the input to insert the Cast node.
        to (int): the data type of the Cast node.

    Note:
        You should pass in a `Rewriter` object if it is called inside a `rewrite`
        method.
    """

    if index < 0:
        index += len(node.input)
    assert index <= len(node.input)
    assert to != TensorProto.UNDEFINED
    cast_node = make_node(
        "Cast",
        inputs=[node.input[index]],
        outputs=[f"{node.name}/cast_i{index}_output0"],
        name=f"{node.name}/cast_i{index}",
        to=to,
    )
    node.input[index] = cast_node.output[0]
    return cast_node


def cast_out(node: NodeProto, index: int, to: int) -> NodeProto:
    """Insert a Cast node after the index-th output of the node.

    Args:
        node (NodeProto): specify a node to insert a Cast node.
        index (int): the index of the output to insert the Cast node.
        to (int): the data type of the Cast node.

    Note:
        You should pass in a `Rewriter` object if it is called inside a `rewrite`
        method.
    """

    if index < 0:
        index += len(node.output)
    assert index <= len(node.output)
    assert to != TensorProto.UNDEFINED
    cast_node = make_node(
        "Cast",
        inputs=[f"{node.name}/cast_o{index}_input0"],
        outputs=[node.output[index]],
        name=f"{node.name}/cast_o{index}",
        to=to,
    )
    node.output[index] = cast_node.input[0]
    return cast_node
