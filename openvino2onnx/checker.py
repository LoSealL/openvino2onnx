"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from typing import Dict, Literal, Optional

import numpy as np
import onnx

from .evaluator import Evaluator
from .graph import OnnxGraph


def show_difference(
    a: np.ndarray, b: np.ndarray, atol: float = 1e-8, rtol: Optional[float] = None
):
    """Print the different element from array a and b."""
    if rtol is not None:
        mask = np.abs(1 - b / a) >= rtol
    mask = np.abs(a - b) >= atol
    return f"a={a[mask]}, b={b[mask]}"


def check_accuracy(
    model1: str | onnx.ModelProto,
    model2: str | onnx.ModelProto,
    input_maps: Optional[Dict[str, np.ndarray]] = None,
    backend: Literal["onnx", "onnxruntime", "openvino"] = "onnx",
) -> Dict[str, float]:
    """
    Check the accuracy of two ONNX models.

    Args:
        model1 (onnx.ModelProto): The first ONNX model to be compared.
        model2 (onnx.ModelProto): The second ONNX model to be compared.
        input_maps (Dict[str, numpy.ndarray], optional): The input data to be used for
            the comparison. If not provided, the model will be run with random input
            data.
        backend (Literal["onnx", "onnxruntime", "openvino"], optional): The backend
            to be used for the comparison. Defaults to "onnx".

    Returns:
        Dict[str, float]: A dictionary containing the accuracy metrics.
    """

    if not isinstance(model1, onnx.ModelProto):
        model1 = onnx.load_model(model1)
    if not isinstance(model2, onnx.ModelProto):
        model2 = onnx.load_model(model2)
    graph1 = OnnxGraph(model1)
    graph2 = OnnxGraph(model2)

    if input_maps is None:
        input_maps = {}
        for input_name in graph1.inputs:
            assert input_name in graph2.inputs
            shape, etype = graph1.tensor_info(input_name)
            dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[etype]
            input_maps[input_name] = np.random.rand(*shape).astype(dtype)
    output_maps = graph1.outputs
    for output_name in graph2.outputs:
        assert output_name in output_maps

    runner1 = Evaluator(model1, backend)
    runner2 = Evaluator(model2, backend)
    results1 = runner1(list(output_maps), input_maps)
    results2 = runner2(list(output_maps), input_maps)

    error_maps = {}
    for name, x, y in zip(output_maps, results1, results2):
        abs_error = np.abs(x - y)
        rel_error = np.abs(x - y) / np.abs(x)
        error_maps[name] = {
            "ABS": abs_error.mean(),
            "REL": rel_error[np.abs(x) > 1e-8].mean(),
        }
    return error_maps
