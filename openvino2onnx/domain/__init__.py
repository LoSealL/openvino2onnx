"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

import os
from pathlib import Path
from typing import List, Optional

import onnx

from .intel import IR_DOMAIN

_LAZY_LOAD = {}


def _make_key(operatorset: onnx.OperatorSetIdProto):
    return (operatorset.domain, operatorset.version)


def _lazy_load_xml_frontend():
    # pylint: disable=import-outside-toplevel
    from .intel.openvino.xml_frontend import (
        openvino_xml_to_onnx_graph as _openvino_xml_to_onnx_graph,
    )

    _LAZY_LOAD[_make_key(IR_DOMAIN)] = _openvino_xml_to_onnx_graph


def openvino_xml_to_onnx_graph(
    xml_path: str | os.PathLike | onnx.ModelProto,
    bin_path: Optional[str | os.PathLike] = None,
):
    """Isolate OpenVINO domain unless a IR xml is detected."""
    key = _make_key(IR_DOMAIN)
    if key in _LAZY_LOAD:
        return _LAZY_LOAD[key](xml_path, bin_path)
    raise ImportError("OpenVINO is not imported yet!")


def detect_domain(
    model: str | os.PathLike | onnx.ModelProto,
) -> List[onnx.OperatorSetIdProto]:
    """Detect a custom domain of a model."""

    if isinstance(model, onnx.ModelProto):
        if len(model.opset_import) == 0:
            return []
        opsets = []
        for opset in model.opset_import:
            if opset.domain not in ("", "ai.onnx", "ai.onnx.ml"):
                opsets.append(opset)
            if opset.domain == IR_DOMAIN.domain:
                _lazy_load_xml_frontend()
        return opsets
    model = Path(model).resolve()
    if model.suffix.lower() == ".xml":
        _lazy_load_xml_frontend()
        return [IR_DOMAIN]
    return []


__all__ = ["openvino_xml_to_onnx_graph", "detect_domain"]
