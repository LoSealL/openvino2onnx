"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

import tempfile
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence
from uuid import uuid4

import networkx as nx
import numpy as np
import onnx
from onnx import numpy_helper
from onnx.helper import make_attribute

from openvino2onnx.evaluator import Evaluator
from openvino2onnx.graph import OnnxGraph

from .logger import debug, error
from .pattern import Pattern
from .utils import attribute_value, canonical_node_name


class RewriterRepeat(Enum):
    """Repeat mode of the rewriter. By default, the rewriter will only match the graph
    once. If the rewriter still wants to match the rewritten graph recursively, set to
    ``INFINITE``.
    """

    ONCE = 1
    INFINITE = 9999  # maximum repeat times


class Rewriter(metaclass=ABCMeta):
    """OnnxGraph rewriter to modify the graph.

    Args:
        pattern (Pattern): a pattern to match nodes
        repeat (RewriterRepeat): repeat mode of the rewriter. Default is ONCE.
    """

    def __init__(self, pattern: Pattern, repeat=RewriterRepeat.ONCE):
        assert isinstance(pattern, Pattern)
        self.pattern = pattern
        repeat_value = repeat.value if isinstance(repeat, RewriterRepeat) else repeat
        self.repeat = min(int(repeat_value), RewriterRepeat.INFINITE.value)
        self.node_to_add = set()
        self.node_to_remove = set()
        self.post_hooks: Dict[int, Callable[[OnnxGraph], OnnxGraph]] = {}

    @abstractmethod
    def rewrite(self, graph: OnnxGraph, nodes: List[onnx.NodeProto], *args, **kwargs):
        """Implement how to rewrite matched nodes in the graph

        Args:
            graph (OnnxGraph): an onnx graph
            nodes (List[NodeProto]): a list of matched nodes
        """

    def add(self, nodes: Sequence[onnx.NodeProto] | onnx.NodeProto):
        """Append nodes to be added to the graph

        Args:
            nodes: a single node or a list of nodes to add
        """
        if isinstance(nodes, onnx.NodeProto):
            self.node_to_add.add(nodes)
        elif isinstance(nodes, Sequence):
            assert all(isinstance(i, onnx.NodeProto) for i in nodes)
            self.node_to_add.update(nodes)
        else:
            raise TypeError(
                f"Expect to add a node or a list of nodes, but got {type(nodes)}."
            )

    def remove(self, nodes: Sequence[onnx.NodeProto] | onnx.NodeProto):
        """Remove nodes from the graph.

        Args:
            nodes: a single node or a list of nodes to remove
        """
        if isinstance(nodes, onnx.NodeProto):
            self.node_to_remove.add(nodes)
        elif isinstance(nodes, Sequence):
            assert all(isinstance(i, onnx.NodeProto) for i in nodes)
            self.node_to_remove.update(nodes)
        else:
            raise TypeError(
                f"Expect to add a node or a list of nodes, but got {type(nodes)}."
            )

    def match_and_rewrite(self, graph: OnnxGraph, *args, **kwargs) -> OnnxGraph:
        """Look up for matched patterns in the graph and rewrite it.

        Args:
            graph (OnnxGraph): onnx graph

        Returns:
            OnnxGraph: rewritten graph
        """
        repeat = self.repeat
        while repeat > 0:
            matched_nodes = self.pattern.match(graph)
            if matched_nodes is None:
                return graph  # no nothing
            if not isinstance(matched_nodes, Generator):
                raise RuntimeError(
                    f"{type(self.pattern)} match function is invalid! It should yield "
                    f"matched nodes, but it returns {type(matched_nodes)}"
                )
            # pylint: disable=attribute-defined-outside-init
            self._graph = graph
            i = None
            for i, nodes in enumerate(matched_nodes):
                if isinstance(nodes, onnx.NodeProto):
                    nodes = [nodes]
                if self.node_to_remove:
                    # filter nodes that been removed in previous pass
                    nodes = list(filter(lambda n: n not in self.node_to_remove, nodes))
                if nodes:
                    try:
                        self.rewrite(graph, nodes, *args, **kwargs)
                    except Exception:
                        nodes_name = ",".join(n.name for n in nodes)
                        error(f"Rewrite nodes [{nodes_name}] failed.")
                        raise
            if i is None:
                return graph  # no match found
            for node in self.node_to_add:
                graph.add_onnx_node(node)
            added_node_names = set([n.name for n in self.node_to_add])
            for node in self.node_to_remove:
                if node.name in added_node_names:
                    # this node has been replaced by a new node, skip it
                    continue
                graph.remove_onnx_node(node)
            self.node_to_add.clear()
            self.node_to_remove.clear()
            repeat -= 1
        for hook_fn in self.post_hooks.values():
            graph = hook_fn(graph)
        return graph

    def get_input_node(
        self, node: onnx.NodeProto, i_or_s: int | str
    ) -> onnx.NodeProto | None:
        """Get the input node to the i-th input."""
        graph = self.graph
        # pylint: disable=protected-access
        if isinstance(i_or_s, int):
            i = i_or_s
            if i < 0:
                i += len(node.input)
            assert i < len(node.input), f"index {i} exceeds input number"
            if name := graph._out_to_node.get(node.input[i]):
                return graph.nodes[name]["pb"]
        else:
            name = i_or_s
            if name := graph._out_to_node.get(name):
                return graph.nodes[name]["pb"]

    def get_input_nodes(self, node: onnx.NodeProto) -> List[onnx.NodeProto | None]:
        """Get all input nodes for the given node."""
        return [self.get_input_node(node, i) for i in node.input]

    def get_output_node(
        self, node: onnx.NodeProto, i_or_s: int | str = 0
    ) -> List[onnx.NodeProto]:
        """Get the output node from the i-th output."""
        graph = self.graph
        if isinstance(i_or_s, int):
            i = i_or_s
            if i < 0:
                i += len(node.output)
            assert i < len(node.output), f"index {i} exceeds output number"
            port = node.output[i]
        else:
            port = i_or_s
        return [s for s in graph.onnx_successors(node) if port in s.input]

    def get_attribute(self, node: onnx.NodeProto, name: str):
        """Try to get the value of an attribute of the node.

        Args:
            node: a node
            name: name of the attribute
        """
        for attr in node.attribute:
            if attr.name == name:
                return attribute_value(attr)

    def set_attribute(self, node: onnx.NodeProto, name: str, value: Any):
        """Set a new value to an attribute of the node."""
        for attr in node.attribute:
            if attr.name == name:
                match attr.type:
                    case onnx.AttributeProto.FLOAT:
                        attr.f = float(value)
                    case onnx.AttributeProto.INT:
                        attr.i = int(value)
                    case onnx.AttributeProto.STRING:
                        attr.s = str(value).encode()
                    case onnx.AttributeProto.TENSOR:
                        attr.t.CopyFrom(numpy_helper.from_array(value))
                    case onnx.AttributeProto.TYPE_PROTO:
                        attr.tp.CopyFrom(value)
                    case onnx.AttributeProto.FLOATS:
                        attr.ClearField("floats")
                        attr.floats.extend(value)
                    case onnx.AttributeProto.INTS:
                        attr.ClearField("ints")
                        attr.ints.extend(value)
                    case onnx.AttributeProto.STRINGS:
                        attr.ClearField("strings")
                        attr.strings.extend(i.encode() for i in value)
                    case onnx.AttributeProto.TENSORS:
                        attr.ClearField("tensors")
                        attr.tensors.extend(value)
                    case onnx.AttributeProto.GRAPHS:
                        attr.ClearField("graphs")
                        attr.graphs.extend(value)
                    case onnx.AttributeProto.TYPE_PROTOS:
                        attr.ClearField("type_protos")
                        attr.type_protos.extend(value)
                    case _:
                        raise ValueError(f"Unsupported attribute type: {attr.type}")
                return
        node.attribute.append(make_attribute(name, value))

    def remove_attribute(self, node: onnx.NodeProto, name: str):
        """Remove an attribute from the node."""
        for attr in list(node.attribute):
            if attr.name == name:
                node.attribute.remove(attr)
                break

    def get_value(self, node: onnx.NodeProto | str, output_name: Optional[str] = None):
        """Get value from a constant node."""
        if isinstance(node, str):
            for init in self.graph.initializer:
                if init.name == node:
                    return numpy_helper.to_array(init)
            # value not found in initializer, try to find it from constant.
            # pylint: disable=protected-access
            cst_node_name = self.graph._out_to_node.get(node)
            if not cst_node_name:
                return
            return self.get_value(self.graph.nodes[cst_node_name]["pb"], node)
        elif node.op_type == "Constant":
            return numpy_helper.to_array(node.attribute[0].t)
        elif node.op_type == "Shape":
            return np.asarray(self.graph.tensor_shape(node.input[0]))
        # try to fold all upstreaming nodes
        all_preds = nx.traversal.dfs_predecessors(self.graph.reverse(), node.name)
        all_preds = set(all_preds)
        all_preds.add(node.name)  # include self
        h = deepcopy(self.graph.onnx_subgraph(all_preds))
        try:
            # pylint: disable=import-outside-toplevel
            from openvino2onnx import convert_graph

            sub_model = h.model
            # extend current value_info due to ``h.model`` cleared it.
            # it's needed because shape inference can not work for custom domains.
            # pylint: disable=protected-access
            sub_model.graph.value_info.extend(h._value_info)
            sub_model = convert_graph(sub_model, [], print_passes=False).model
        except Exception:  # pylint: disable=broad-except
            debug(f"convert subgraph failed on {node.name}: [{all_preds}]")
            return
        try:
            if output_name is None:
                output_name = sub_model.graph.output[0].name
            runner = Evaluator(sub_model)
            return runner([output_name], {})[0]
        except Exception:  # pylint: disable=broad-except
            node_name = canonical_node_name(node.name)
            temp_save = Path(tempfile.gettempdir()) / f"{node_name}.onnx"
            onnx.save_model(sub_model, temp_save)
            debug(f"evaluate subgraph failed on {node.name}: {temp_save}")
            return

    def register_post_hook(self, hook: Callable[[OnnxGraph], OnnxGraph]) -> int:
        """Register a post-rewrite hook function.

        A hook function must be a callable that takes an OnnxGraph as input and returns
        an OnnxGraph as output. It will be called after the last run of rewrite.

        Args:
            hook: a hook function to be registered

        Returns:
            int: a unique id of the hook function
        """
        if not callable(hook):
            raise TypeError(f"Expect a callable, but got {type(hook)}")
        uid = uuid4().int
        self.post_hooks[uid] = hook
        return uid

    def remove_post_hook(self, uid: int):
        """Remove a post-rewrite hook function.

        Args:
            uid: a unique id of the hook function to be removed
        """
        self.post_hooks.pop(uid)

    @property
    def graph(self) -> OnnxGraph:
        """Get the current graph."""
        return self._graph

    def __call__(self, graph: OnnxGraph, *args, **kwargs) -> OnnxGraph:
        """Make rewriter a callable."""
        return self.match_and_rewrite(graph, *args, **kwargs)

    def __iadd__(self, nodes: Sequence[onnx.NodeProto] | onnx.NodeProto):
        self.add(nodes)
        return self

    def __isub__(self, nodes: Sequence[onnx.NodeProto] | onnx.NodeProto):
        self.remove(nodes)
        return self
