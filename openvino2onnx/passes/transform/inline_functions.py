"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

# pylint: disable=arguments-differ

import json
from copy import deepcopy
from typing import List
from uuid import uuid4

from onnx.onnx_pb import NodeProto

from openvino2onnx.graph import OnnxGraph
from openvino2onnx.passes import PASSES
from openvino2onnx.passes.pattern import SingleNodePattern
from openvino2onnx.passes.rewriter import Rewriter


@PASSES.register("inline_functions")
class InlineFunctionsRewriter(Rewriter):
    """This pass inlines all functions into the main graph."""

    def __init__(self):
        super().__init__(SingleNodePattern().with_domain("*"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto], force: bool = False):
        node = nodes[0]
        if node.op_type not in graph.functions:
            raise RuntimeError(f"function {node.op_type} not found in the graph")

        try:
            node_names = json.loads(node.doc_string)
            assert isinstance(node_names, list)
            assert all(map(lambda x: isinstance(x, str), node_names))
        except Exception:
            node_names = []
        if len(node_names) == 0 and not force:
            return

        tag = f"{node.name}/{node.op_type}"

        # step 1: find the input and output port that matches the custom node
        func = graph.functions[node.op_type]
        func_nodes = [deepcopy(node) for node in func.node]
        assert len(func_nodes) == len(node_names) or len(node_names) == 0

        for i, func_input in enumerate(func.input):
            input_name = node.input[i]
            for n in func_nodes:
                for j, node_input in enumerate(n.input):
                    if node_input == func_input:
                        n.input[j] = input_name
                    else:
                        n.input[j] = f"{tag}/{n.input[j]}"
        for i, func_output in enumerate(func.output):
            output_name = node.output[i]
            for n in func_nodes:
                for j, node_output in enumerate(n.output):
                    if node_output == func_output:
                        n.output[j] = output_name
                    else:
                        n.output[j] = f"{tag}/{n.output[j]}"
        # step 2: make a copy of each function node appended with identity names
        for i, n in enumerate(func_nodes):
            if not n.name:
                n.name = f"{uuid4().hex}"
            n.name = node_names[i] if node_names else f"{tag}/{n.name}"

        # step 3: assign attributes
        for attr in node.attribute:
            value = self.get_attribute(node, attr.name)
            for n in func_nodes:
                for attr2 in n.attribute:
                    if attr2.ref_attr_name == attr.name:
                        self.set_attribute(n, attr2.name, value)

        self += func_nodes
        self -= node
