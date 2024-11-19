"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

import os
import warnings
from copy import deepcopy
from itertools import chain
from pathlib import Path
from typing import IO, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

import networkx as nx
import onnx
from onnx.helper import (
    make_graph,
    make_model,
    make_operatorsetid,
    make_tensor_type_proto,
    make_value_info,
)


class OnnxGraph(nx.DiGraph):
    """Create a DAG from onnx graph."""

    def __init__(self, model: Optional[onnx.ModelProto] = None) -> None:
        if model is None:
            super().__init__()
        else:
            assert isinstance(model, onnx.ModelProto)
            super().__init__(name=model.graph.name)
            self._model = model
            self._node_to_out = {}
            self._out_to_node = {}
            self._functions = {f.name: f for f in model.functions}
            try:
                graph = onnx.shape_inference.infer_shapes(model).graph
            except onnx.shape_inference.InferenceError as ex:
                warnings.warn(f"Inferring shape failed, value info is not set:\n {ex}")
                graph = model.graph
            self.inputs = {i.name: n for n, i in enumerate(graph.input)}
            self.outputs = {i.name: n for n, i in enumerate(graph.output)}
            self._value_info = graph.value_info
            for node in graph.node:
                self._add_onnx_node_internal(node)
            self._build_edges()

    def _add_onnx_node_internal(self, node: onnx.NodeProto) -> None:
        if not node.HasField("name"):
            node.name = f"{node.op_type}/{uuid4().hex}"
        name = node.name
        has_input = any(i in self.inputs for i in node.input)
        has_output = any(i in self.outputs for i in node.output)
        # clear out-dated edges on this node
        if name in self:
            self.remove_node(name)
        self.add_node(name, pb=node, has_input=has_input, has_output=has_output)
        self._node_to_out[name] = node.output

    def _build_edges(self):
        out_to_node = {i: k for k, v in self._node_to_out.items() for i in v}
        for n in self.nodes:
            node = self.nodes[n]["pb"]
            for i in node.input:
                if upstream := out_to_node.get(i):
                    self.add_edge(upstream, n)
        self._out_to_node = out_to_node

    def add_onnx_node(self, node: onnx.NodeProto) -> None:
        """Insert a node and its edges."""
        assert isinstance(node, onnx.NodeProto)
        self._add_onnx_node_internal(node)
        self._build_edges()
        # remove connected edge from inputs
        for output_name in node.output:
            for i, graph_input in enumerate(self.input):
                if graph_input.name == output_name:
                    self.input.pop(i)
                    self.inputs.pop(output_name)

    def remove_onnx_node(self, node: onnx.NodeProto | str, no_replace=False) -> None:
        """Remove a node from the graph"""
        assert isinstance(node, (onnx.NodeProto, str))
        if isinstance(node, str):
            name = node
            node = self.nodes[name]["pb"]
        else:
            name = node.name
        if name not in self:
            raise ValueError(f"Node {name} is not existing in the graph!")
        assert isinstance(node, onnx.NodeProto)
        if no_replace:
            # dump node output info to inputs
            for output_name in node.output:
                if self._out_to_node.get(output_name) != name:
                    # node is replaced
                    continue
                shape, dtype = self.tensor_info(output_name)
                self.input.append(
                    make_value_info(output_name, make_tensor_type_proto(dtype, shape))
                )
                self.inputs[output_name] = len(self.input) - 1
        self._node_to_out.pop(name)
        for output_name in node.output:
            if self._out_to_node[output_name] == name:
                self._out_to_node.pop(output_name)
                # output is not claimed by added node, remove it from graph
                if output_name in self.outputs:
                    self.remove_output(output_name)
        self.remove_node(name)

    def onnx_predecessors(self, n: onnx.NodeProto | str) -> List[onnx.NodeProto]:
        """Returns a list of predecessor nodes of n."""
        assert isinstance(n, (onnx.NodeProto, str))
        preds = self.predecessors(n if isinstance(n, str) else n.name)
        return [self.nodes[p]["pb"] for p in preds]

    def onnx_successors(self, n: onnx.NodeProto | str) -> List[onnx.NodeProto]:
        """Returns a list of successors nodes of n."""
        assert isinstance(n, (onnx.NodeProto, str))
        succs = self.successors(n if isinstance(n, str) else n.name)
        return [self.nodes[s]["pb"] for s in succs]

    def onnx_subgraph(self, nodes: Iterable[onnx.NodeProto | str]) -> "OnnxGraph":
        """Create a sub onnx graph from nodes"""
        sub: nx.DiGraph = self.subgraph(
            (n if isinstance(n, str) else n.name for n in nodes)
        )
        subonnx = make_graph([], f"subgraph of {self.name}", inputs=[], outputs=[])
        # inherite initializers and value infos
        initializer_map = {i.name: i for i in self.initializer}
        subonnx.value_info.extend(self._value_info)

        def _compute_input(node):
            check_list = set(initializer_map.keys())
            for pred in sub.predecessors(node.name):
                check_list.update(sub.nodes[pred]["pb"].output)
            for input_name in node.input:
                if input_name and input_name not in check_list:
                    yield input_name
                if input_name in initializer_map:
                    subonnx.initializer.append(initializer_map[input_name])
                    initializer_map.pop(input_name)  # avoid duplicated initializer

        graph_inputs = set()
        for i in nx.topological_sort(sub):
            node = self.nodes[i]["pb"]
            subonnx.node.append(node)
            for output_name in node.output:
                is_outside = any(  # has an edge connected outside subgraph
                    output_name in j.input
                    for j in self.onnx_successors(node)
                    if j.name not in sub
                )
                if sub.out_degree(i) == 0 or is_outside:
                    shape, dtype = self.tensor_info(output_name)
                    subonnx.output.append(
                        make_value_info(
                            output_name, make_tensor_type_proto(dtype, shape)
                        )
                    )
            node_inputs = list(_compute_input(node))
            if node.op_type != "Constant" and node_inputs:
                graph_inputs.update(node_inputs)
        for input_name in graph_inputs:
            shape, dtype = self.tensor_info(input_name)
            subonnx.input.append(
                make_value_info(input_name, make_tensor_type_proto(dtype, shape))
            )
        sub_model = make_model(
            subonnx,
            doc_string=self._model.doc_string,
            domain=self._model.domain,
            model_version=self._model.model_version,
            ir_version=self._model.ir_version,
            opset_imports=self._model.opset_import,
            producer_name=self._model.producer_name,
            producer_version=self._model.producer_version,
            functions=self.functions.values(),
        )
        return OnnxGraph(sub_model)

    def onnx_add_function(self, func: onnx.FunctionProto) -> None:
        """Add a function to the graph.

        Args:
            func (onnx.FunctionProto): an onnx function proto.
        """
        if func.name not in self._functions:
            self._functions[func.name] = func
            self._model.opset_import.extend(func.opset_import)

    def tensor_info(self, name: str) -> Tuple[List[int | str] | None, int]:
        """Get shape and dtype of a tensor by its name."""
        shape: Optional[List[int | str]] = None
        dtype = onnx.TensorProto.UNDEFINED
        if name in self.inputs or name in self.outputs:
            value_infos = list(chain(self.input, self.output))
        else:
            value_infos = self._value_info
        for value_info in value_infos:
            if value_info.name == name:
                dtype = value_info.type.tensor_type.elem_type
                if not value_info.type.tensor_type.HasField("shape"):
                    break  # no shape info, skip
                dims = [i for i in value_info.type.tensor_type.shape.dim]
                shape = [0 for _ in dims]
                for i, s in enumerate(dims):
                    if s.dim_param and s.dim_value == 0:
                        shape[i] = s.dim_param
                    else:
                        shape[i] = int(s.dim_value)
                break
        if shape is None:
            for tensor in self.initializer:
                if tensor.name == name:
                    shape = [int(i) for i in tensor.dims]
                    dtype = tensor.data_type
        return shape, dtype

    def tensor_shape(self, name: str) -> List[int | str]:
        """Get shape from a given tensor name."""
        shape, _ = self.tensor_info(name)
        if shape is None:
            raise ValueError(f"Can't find tensor shape with name '{name}'")
        return shape

    def tensor_type(self, name: str) -> int:
        """Get tensor type from a given tensor name."""
        _, elem_type = self.tensor_info(name)
        if elem_type == onnx.TensorProto.UNDEFINED:
            raise ValueError(f"Can't find tensor dtype with name '{name}'")
        return elem_type

    def set_input(self, node: onnx.NodeProto | str, input_name: str):
        """Set the `input_name` of `node` as the graph input."""
        if isinstance(node, str):
            node = self.nodes[node]["pb"]
        assert isinstance(node, onnx.NodeProto)
        if input_name not in node.input:
            raise ValueError(f"Can't find {input_name} in {node.name}")
        if input_name in self.inputs:
            return  # already input
        shape, dtype = self.tensor_info(input_name)
        self.inputs[input_name] = len(self.input)
        if shape is None or dtype is None:
            raise ValueError(f"Can't find tensor shape and dtype of {input_name}")
        self.input.append(
            make_value_info(input_name, make_tensor_type_proto(dtype, shape))
        )

    def set_output(self, node: onnx.NodeProto | str, output_name: str):
        """Set the `output_name` of `node` as the graph output."""
        if isinstance(node, str):
            node = self.nodes[node]["pb"]
        assert isinstance(node, onnx.NodeProto)
        if output_name not in node.output:
            raise ValueError(f"Can't find {output_name} in {node.name}")
        if output_name in self.outputs:
            return  # already output
        shape, dtype = self.tensor_info(output_name)
        self.outputs[output_name] = len(self.output)
        if shape is None or dtype is None:
            raise ValueError(f"Can't find tensor shape and dtype of {output_name}")
        self.output.append(
            make_value_info(output_name, make_tensor_type_proto(dtype, shape))
        )

    def remove_input(self, input_name: str):
        """Remove the `input_name` from the graph input."""
        ind = self.inputs.pop(input_name)
        self.input.pop(ind)
        for i in self.inputs:
            if self.inputs[i] > ind:
                self.inputs[i] -= 1

    def remove_output(self, output_name: str):
        """Remove the `output_name` from the graph output."""
        ind = self.outputs.pop(output_name)
        self.output.pop(ind)
        for i in self.outputs:
            if self.outputs[i] > ind:
                self.outputs[i] -= 1

    def rename_input(self, old_name: str, new_name: str):
        """Rename the `old_name` input to `new_name`."""
        self.inputs[new_name] = self.inputs.pop(old_name)
        self.input[self.inputs[new_name]].name = new_name
        for n in nx.topological_sort(self):
            if self.in_degree(n) > 0:
                break
            input_node = self.nodes[n]["pb"]
            for ind, i in enumerate(input_node.input):
                if i == old_name:
                    input_node.input[ind] = new_name

    def rename_output(self, old_name: str, new_name: str):
        """Rename the `old_name` output to `new_name`."""
        self.outputs[new_name] = self.outputs.pop(old_name)
        self.output[self.outputs[new_name]].name = new_name
        self._out_to_node[new_name] = self._out_to_node.pop(old_name)
        output_node = self._out_to_node[new_name]
        output_node = self.nodes[output_node]["pb"]
        for ind, i in enumerate(output_node.output):
            if i == old_name:
                output_node.output[ind] = new_name
        self._node_to_out[output_node.name] = output_node.output

    @property
    def input(self):
        """Return a list of graph inputs value info."""
        return self._model.graph.input

    @property
    def output(self):
        """Return a list of graph outputs value info."""
        return self._model.graph.output

    @property
    def initializer(self):
        """Return a list of graph initializer tensors."""
        return self._model.graph.initializer

    @property
    def model_version(self) -> int:
        """Return the model version of the model."""
        return self._model.model_version

    @property
    def ir_version(self) -> int:
        """Return the ir version of the model."""
        return self._model.ir_version

    @property
    def opset_version(self) -> int:
        """Return the opset version of the model."""
        return max(
            i.version for i in self._model.opset_import if i.domain in ("", "ai.onnx")
        )

    @property
    def functions(self) -> Dict[str, onnx.FunctionProto]:
        """Return the functions inside the model."""
        return self._functions

    @opset_version.setter
    def opset_version(self, version: int):
        """Set the opset version of the model."""
        # keep only one ai.onnx opset
        for opset in list(self._model.opset_import):
            if opset.domain in ("", "ai.onnx"):
                self._model.opset_import.remove(opset)
        self._model.opset_import.append(make_operatorsetid("", version))

    @property
    def model(self) -> onnx.ModelProto:
        """Make new model"""
        graph = deepcopy(self._model.graph)
        graph.ClearField("node")
        graph.ClearField("value_info")
        for n in nx.topological_sort(self):
            graph.node.append(self.nodes[n]["pb"])
        model = make_model(
            graph,
            doc_string=self._model.doc_string,
            domain=self._model.domain,
            model_version=self._model.model_version,
            ir_version=self._model.ir_version,
            opset_imports=self._model.opset_import,
            producer_name=self._model.producer_name,
            producer_version=self._model.producer_version,
            functions=self._functions.values(),
        )
        model.metadata_props.extend(self._model.metadata_props)
        return model

    # pylint: disable=redefined-builtin
    def save(
        self,
        model_path: str | os.PathLike | IO[bytes],
        format: Optional[str] = None,
        infer_shapes: bool = True,
        check: bool = True,
    ):
        """Serialize the graph to onnx model and save to model_path."""
        model = self.model
        if not isinstance(model_path, (str, os.PathLike)):
            format = "protobuf"
        elif format == "protobuf" or format is None:
            model_path = Path(model_path).with_suffix(".onnx")
        elif format == "textproto":
            model_path = Path(model_path).with_suffix(".pbtxt")
        elif format == "json":
            model_path = Path(model_path).with_suffix(".json")
        elif format == "onnxtxt":
            model_path = Path(model_path).with_suffix(".onnxtxt")
        if infer_shapes:
            model = onnx.shape_inference.infer_shapes(model, data_prop=True)
        if check:
            onnx.checker.check_model(model, full_check=True)
        onnx.save_model(model, model_path, format=format)
