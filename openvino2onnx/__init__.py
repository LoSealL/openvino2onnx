"""An Open Neural Network Exchange (ONNX) Optimization and Transformation Tool.

Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

"""

__version__ = "1.0.1"

import os
from copy import deepcopy
from typing import Literal, Optional, Sequence

import onnx
from onnx import ModelProto
from onnx.helper import make_operatorsetid

from .domain import IR_DOMAIN, detect_domain, openvino_xml_to_onnx_graph
from .graph import OnnxGraph
from .pass_manager import PassManager
from .passes.convert.version_converter.downgrade import downgrade_op_version
from .passes.convert.version_converter.upgrade import upgrade_op_version


def convert_graph(
    model: str | os.PathLike | ModelProto,
    passes: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
    onnx_format: Optional[Literal["protobuf", "textproto", "json", "onnxtxt"]] = None,
    strict: bool = False,
    configs: Optional[dict] = None,
    print_passes: bool = True,
    target_opset: Optional[int] = None,
) -> OnnxGraph:
    """Convert an ONNX model to OnnxGraph

    Args:
        model (str | os.PathLike | ModelProto): path to the model or a loaded model.
        passes (Sequence[str], optional): Names of selected passes. Defaults to None.
        exclude (Sequence[str], optional): Names of excluded passes. Defaults to None.
        onnx_format (str, optional): The serialization format of model file.
        strict (bool, optional): Break if any pass goes wrong. Defaults to False.
        configs (dict, optional): Specify configuration for passes
        print_passes (bool, optional): Print the selected passes. Defaults to True.
        target_opset (int, optional): Target opset version for ONNX domain. Defaults
            to ``OPENVINO2ONNX_OPSET.version``.

    Returns:
        OnnxGraph: converted graph
    """
    for opset in detect_domain(model):
        if opset.domain == IR_DOMAIN.domain and opset.version <= IR_DOMAIN.version:
            model = openvino_xml_to_onnx_graph(model)
    if isinstance(model, (str, os.PathLike)):
        model = onnx.load_model(model, format=onnx_format)
    else:
        model = deepcopy(model)
    graph = OnnxGraph(model)
    if target_opset is None:
        # align models to opset v19 because all passes is designed under opset19
        graph = upgrade_op_version(graph, op_version=19)
    pm = PassManager(passes, exclude=exclude, configs=configs)
    if print_passes:
        print(pm)
    graph = pm.optimize(graph, strict=strict)
    if target_opset is not None:
        if target_opset < graph.opset_version:
            graph = downgrade_op_version(graph, target_opset)
        else:
            graph = upgrade_op_version(graph, target_opset)
    return graph


def convert(
    model: str | os.PathLike | ModelProto,
    passes: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
    onnx_format: Optional[Literal["protobuf", "textproto", "json", "onnxtxt"]] = None,
    strict: bool = False,
    configs: Optional[dict] = None,
    print_passes: bool = True,
    target_opset: Optional[int] = None,
) -> ModelProto:
    """Convert an ONNX model with default or given passes

    Args:
        model (str | os.PathLike | ModelProto): path to the model or a loaded model.
        passes (Sequence[str], optional): Names of selected passes. Defaults to None.
        exclude (Sequence[str], optional): Names of excluded passes. Defaults to None.
        onnx_format (str, optional): The serialization format of model file.
        strict (bool, optional): Break if any pass goes wrong. Defaults to False.
        configs (dict, optional): Specify configuration for passes.
        print_passes (bool, optional): Print the selected passes. Defaults to True.
        target_opset (int, optional): Target opset version for ONNX domain. Defaults
            to ``OPENVINO2ONNX_OPSET.version``.
    """

    graph = convert_graph(
        model=model,
        passes=passes,
        exclude=exclude,
        onnx_format=onnx_format,
        strict=strict,
        configs=configs,
        print_passes=print_passes,
        target_opset=target_opset,
    )
    return graph.model


__all__ = ["convert", "PassManager"]

# make NodeProto hashable using node name
onnx.NodeProto.__hash__ = lambda self: hash(self.name)  # type: ignore

OPENVINO2ONNX_IR_VERSION = onnx.IR_VERSION_2023_5_5
"""Currently used IR version, since most runtime supports up to this version."""

OPENVINO2ONNX_OPSET = make_operatorsetid("", 19)
"""Currently used opset version, since most runtime supports up to this version."""
