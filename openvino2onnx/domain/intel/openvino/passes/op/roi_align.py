"""
Copyright Wenyi Tang 2024-2025

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from openvino2onnx.graph import OnnxGraph

from . import OP_CONVERT, BaseNodeConversion


@OP_CONVERT.register(name="roi_align")
class ROIAlign(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/detection/roi-align-9.html

    https://onnx.ai/onnx/operators/onnx__RoiAlign.html
    """

    _aligned_mode_map = {
        "half_pixel_for_nn": "half_pixel",
        "asymmetric": "output_half_pixel",
    }

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        pooled_h = self.get_attribute(ori_node, "pooled_h")
        assert isinstance(pooled_h, (int, float, str))
        pooled_w = self.get_attribute(ori_node, "pooled_w")
        assert isinstance(pooled_w, (int, float, str))
        sampling_ratio = self.get_attribute(ori_node, "sampling_ratio")
        assert isinstance(sampling_ratio, (int, float, str))
        spatial_scale = self.get_attribute(ori_node, "spatial_scale")
        assert isinstance(spatial_scale, (int, float, str))
        mode = self.get_attribute(ori_node, "mode")
        aligned_mode = self.get_attribute(ori_node, "aligned_mode") or "asymmetric"
        if aligned_mode not in self._aligned_mode_map:
            raise ValueError(
                "aligned_mode=half_pixel is not supported."
                f" Accepted mode are {self._aligned_mode_map.keys()}"
            )
        return make_node(
            "RoiAlign",
            inputs=ori_node.input,
            outputs=ori_node.output,
            name=ori_node.name,
            output_height=int(pooled_h),
            output_width=int(pooled_w),
            sampling_ratio=int(sampling_ratio),
            spatial_scale=float(spatial_scale),
            mode=mode,
            coordinate_transformation_mode=self._aligned_mode_map[aligned_mode],
        )
