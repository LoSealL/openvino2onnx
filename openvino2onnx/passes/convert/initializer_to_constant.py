"""
Copyright Wenyi Tang 2024-2025

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from onnx.helper import make_tensor_type_proto, make_value_info
from onnx.numpy_helper import from_array, to_array

from openvino2onnx.graph import OnnxGraph
from openvino2onnx.passes import PASSES
from openvino2onnx.passes.utils import make_constant


@PASSES.register()
def eliminate_initializer_input(graph: OnnxGraph) -> OnnxGraph:
    """Eliminate graph input that is actually an initializer"""
    pop_list = []
    for init in graph.initializer:
        if init.name in graph.inputs:
            i = graph.inputs.pop(init.name)
            pop_list.append(i)
    for i in sorted(pop_list, reverse=True):
        graph.input.pop(i)

    # re-assign index of inputs.
    for i, node in enumerate(graph.input):
        graph.inputs[node.name] = i
    return graph


@PASSES.register(deps=["initializer_unique", "eliminate_initializer_input"])
def initializer_to_constant(graph: OnnxGraph) -> OnnxGraph:
    """Convert initializer value to node Constant."""
    for init in graph.initializer:
        node = make_constant(init.name, to_array(init))
        if node.name in graph:
            # initializer could have same name as other node,
            # rename it to avoid conflict.
            node.name += "_init"
        node.output[0] = init.name
        graph._value_info.append(  # pylint: disable=W0212
            make_value_info(
                node.output[0],
                make_tensor_type_proto(init.data_type, init.dims),
            )
        )
        graph.add_onnx_node(node)
    while graph.initializer:
        graph.initializer.pop()
    return graph


@PASSES.register(name="eliminate_duplicated_initializer")
def eliminate_duplicated_initializer(graph: OnnxGraph) -> OnnxGraph:
    """Eliminate duplicated initializer.
    It can be used for initializer_to_constant"""
    init_set = {init.name: init for init in graph.initializer}
    while graph.initializer:
        graph.initializer.pop()
    graph.initializer.extend(init_set.values())
    return graph


@PASSES.register(name="initializer_unique")
def initializer_unique(graph: OnnxGraph) -> OnnxGraph:
    """Makes all initializer unique but rarely needed.
    Generally only eliminate_duplicated_initializer is needed.
    """
    init_name_map = {}
    for init in graph.initializer:
        init_name_map[init.name] = {
            "init": init,
            "times": 0,
        }

    init_add = []
    for node in graph:
        node_pb = graph.nodes[node]["pb"]
        for i, name in enumerate(node_pb.input):
            if name in init_name_map:
                init_name_map[name]["times"] += 1
                if init_name_map[name]["times"] == 1:
                    continue
                new_name = name + f"/{init_name_map[name]['times']}"
                node_pb.input[i] = new_name
                new_init = from_array(to_array(init_name_map[name]["init"]), new_name)
                init_add.append(new_init)

    # should remove duplicated items in initializer
    graph._model.graph.ClearField("initializer")  # pylint: disable=W0212
    graph.initializer.extend(init_add)
    for v in init_name_map.values():
        graph.initializer.append(v["init"])
    return graph
