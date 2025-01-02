"""
Copyright Wenyi Tang 2024-2025

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from typing import Dict, List

from openvino2onnx import OnnxGraph
from openvino2onnx.passes import PASSES


@PASSES.register("reshape_model", deps=["infer_shape"])
def reshape_model(graph: OnnxGraph, shape_info: Dict[str, List[int | str]]):
    """Update model input and output dimensions.

    Args:
        shape_info (Dict[str, List[int | str]]): A dictionary of I/O names and
            their new shapes.

    Example:

        openvino2onnx model.onnx -a reshape_model --config-file config.json

        # config.json:
        {
            "reshape_model:0": {
                "shape_info": {
                    "input": [1, 2, 3, 4],
                    "output": [4, 3, 2, 1]
                }
            }
        }
    """

    def _update_dim(dims, shape):
        for d, i in zip(dims, shape):
            if isinstance(i, int):
                if i < 0 and i != -1:
                    raise ValueError(f"negative value in shape must be -1, but got {i}")
                d.dim_value = i
            elif isinstance(i, str):
                d.dim_param = i
            else:
                raise TypeError(f"shape must be int or str, but got {type(i)}")

    for _, input_info in enumerate(graph.input):
        shape = shape_info[input_info.name]
        _update_dim(input_info.type.tensor_type.shape.dim, shape)
    for _, output_info in enumerate(graph.output):
        shape = shape_info[output_info.name]
        _update_dim(output_info.type.tensor_type.shape.dim, shape)
    return graph


@PASSES.register("reset_model_batch", deps=["infer_shape"])
def reset_model_batch(graph: OnnxGraph, **kwargs):
    """Reset the batch size of the model.

    Batch size is the 1st dimension of all tensors.

    Note:

        This pass does not check the feasibility of the new batch size. And
        it will fail to infer shapes if the new batch size is not compatible
        with the previous shapes.
    """
    kwargs.setdefault("batch", 1)
    kwargs.setdefault("batch_size", 1)
    batch = kwargs.get("batch") or kwargs.get("batch_size")
    assert isinstance(batch, int) and batch > 0
    for _, input_info in enumerate(graph.input):
        input_info.type.tensor_type.shape.dim[0].dim_value = batch
    for _, output_info in enumerate(graph.output):
        output_info.type.tensor_type.shape.dim[0].dim_value = batch
    return graph
