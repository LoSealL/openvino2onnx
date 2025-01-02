"""
Copyright Wenyi Tang 2024-2025

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

# pylint: disable=line-too-long

from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from openvino2onnx.graph import OnnxGraph
from openvino2onnx.passes.utils import make_constant

from . import OP_CONVERT, BaseNodeConversion


class BaseReduce(BaseNodeConversion):
    """Generic implementation of reduce operations."""

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        keepdims = self.get_attribute(ori_node, "keep_dims")
        assert isinstance(keepdims, str)
        keepdims = keepdims.lower() == "true"
        # canonicalize axes to int64
        axes = self.get_value_or_die(ori_node.input[1]).astype("int64")
        if axes.ndim == 0:
            axes = axes[None]  # axes must be a 1D tensor
        axes_node = make_constant(f"{ori_node.name}/axes", axes)
        ori_node.input[1] = axes_node.output[0]
        self += axes_node
        return make_node(
            self.__class__.__name__,
            inputs=ori_node.input,
            outputs=ori_node.output,
            name=ori_node.name,
            keepdims=1 if keepdims else 0,
        )


@OP_CONVERT.register(name="reduce_l1")
class ReduceL1(BaseReduce):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/reduction/reduce-l1-4.html

    https://onnx.ai/onnx/operators/onnx__ReduceL1.html
    """


@OP_CONVERT.register(name="reduce_l2")
class ReduceL2(BaseReduce):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/reduction/reduce-l2-4.html

    https://onnx.ai/onnx/operators/onnx__ReduceL2.html
    """


@OP_CONVERT.register(name="reduce_max")
class ReduceMax(BaseReduce):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/reduction/reduce-max-1.html

    https://onnx.ai/onnx/operators/onnx__ReduceMax.html
    """


@OP_CONVERT.register(name="reduce_mean")
class ReduceMean(BaseReduce):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/reduction/reduce-mean-1.html

    https://onnx.ai/onnx/operators/onnx__ReduceMean.html
    """


@OP_CONVERT.register(name="reduce_min")
class ReduceMin(BaseReduce):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/reduction/reduce-min-1.html

    https://onnx.ai/onnx/operators/onnx__ReduceMin.html
    """


@OP_CONVERT.register(name="reduce_prod")
class ReduceProd(BaseReduce):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/reduction/reduce-prod-1.html

    https://onnx.ai/onnx/operators/onnx__ReduceProd.html
    """


@OP_CONVERT.register(name="reduce_sum")
class ReduceSum(BaseReduce):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/reduction/reduce-sum-1.html

    https://onnx.ai/onnx/operators/onnx__ReduceSum.html
    """
