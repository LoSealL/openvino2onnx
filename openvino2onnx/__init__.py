"""openvino2onnx is a tool to convert openvino IR format to ONNX.
"""

import os
from typing import Optional

import onnx

from .builder import build
from .ir11 import ir_to_graph
from .legalize import legalize

__version__ = "0.2.0"

__all__ = ["build", "ir_to_graph"]


def transform(
    model_url: str | os.PathLike,
    model_bin: Optional[str | os.PathLike] = None,
    force_fp32: bool = True,
    opset_version: int = 17,
) -> onnx.ModelProto:
    """Transform an OpenVINO IR (.xml + .bin) back to a legal ONNX model.

    Args:
        model_url (str | os.PathLike): model url, it is possible to be a local file
            or a remote url or from OpenVINO model zoo with "omz://<model-name>".
        model_bin (Optional[str | os.PathLike], optional): Weights file of the model.
            If not provided, search the .bin file with same name as model .xml file.
            Defaults to None.
        force_fp32 (bool, optional): Force all fp16 data to be fp32. Defaults to True.
        opset_version (int, optional): Export onnx opset version. Defaults to 17.

    Returns:
        onnx.ModelProto: _description_
    """
    assert 11 <= opset_version < 20, "opset version range is [11, 19]"
    try:
        graph = ir_to_graph(model_url, model_bin, force_fp32)
    except (NotImplementedError, OSError, RuntimeError, ValueError):
        print(f"[E] Failed to build {model_url} to graph.")
        raise
    try:
        graph = legalize(graph)
    except RuntimeError:
        print("[E] Failed to legalize the graph.")
        raise
    try:
        model = build(graph, version=opset_version)
    except Exception:  # pylint: disable=W0718
        print(
            "[E] Failed to build graph to onnx model "
            f"v{onnx.IR_VERSION} opset={opset_version}"
        )
        raise
    return model
