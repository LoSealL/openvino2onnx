"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from pathlib import Path
from typing import Optional

import onnx

from openvino2onnx import OPENVINO2ONNX_IR_VERSION, OPENVINO2ONNX_OPSET, PassManager
from openvino2onnx.graph import OnnxGraph

from .ir.ir11 import ir_to_onnx
from .passes.op import IR_PASSES, OP_CONVERT

_POST_PASSES = [
    "conv_dequantize_weight",
    "gemm_dequantize_weight",
    "recalculate_dequantize_weight",
    "eliminate_dead_nodes",
]


def openvino_xml_to_onnx_graph(
    model_path: str | Path | onnx.ModelProto,
    model_bin: Optional[str | Path] = None,
) -> onnx.ModelProto:
    """Convert OpenVINO IR .xml file to an equivalent ONNX graph.

    Args:
        model_path (str | Path): Path to the OpenVINO IR.xml file.
        model_bin (str | Path, optional): Path to the OpenVINO IR.bin file.
            Defaults to None. If model_bin is None, the binary is to be found in the
            same directory as the xml file with the same name but with a.bin extension.
    """
    if isinstance(model_path, onnx.ModelProto) and model_bin is None:
        ov_onnx = model_path  # for nested call but do not expose this interface
    elif not isinstance(model_path, onnx.ModelProto):
        ov_onnx = ir_to_onnx(model_path, model_bin)
    else:
        raise ValueError("model_bin is specified but model_path is given ModelProto")
    graph = OnnxGraph(ov_onnx)
    # sort OP_CONVERT here to make sure to put "Trivial" to the end
    passes = set(IR_PASSES).difference(OP_CONVERT)
    pm = PassManager(list(passes) + sorted(OP_CONVERT) + _POST_PASSES)
    model = pm.optimize(graph, strict=True).model
    model.ir_version = OPENVINO2ONNX_IR_VERSION
    for opset in list(model.opset_import):
        if opset.domain not in ("", "ai.onnx", "ai.onnx.ml"):
            model.opset_import.remove(opset)
    if not model.opset_import:
        model.opset_import.append(OPENVINO2ONNX_OPSET)
    return model
