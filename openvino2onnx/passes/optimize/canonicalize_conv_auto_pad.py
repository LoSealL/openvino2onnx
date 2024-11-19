"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from typing import List

from onnx.onnx_pb import NodeProto

from openvino2onnx.graph import OnnxGraph
from openvino2onnx.passes import PASSES
from openvino2onnx.passes.pattern import SingleNodePattern
from openvino2onnx.passes.rewriter import Rewriter


@PASSES.register(name="canonicalize_conv_autopad")
class CanonicalizeConvAutoPadRewriter(Rewriter):
    """After onnxsim it may combine pad into Conv, but still keep auto_pad attribute.
    This will cause an error in onnxruntime. So we need to remove the auto_pad
    attribute if pads is also set.
    """

    def __init__(self):
        super().__init__(
            SingleNodePattern("Conv").with_attr("auto_pad").with_attr("pads")
        )

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto], *args, **kwargs):
        node = nodes[0]
        auto_pad = self.get_attribute(node, "auto_pad")
        if auto_pad != "NOTSET":
            self.set_attribute(node, "auto_pad", "NOTSET")
