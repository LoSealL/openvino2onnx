"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from contextlib import suppress

from openvino2onnx.graph import OnnxGraph
from openvino2onnx.passes import L3

with suppress(ImportError):
    import onnxsim

    @L3.register(
        patch=[
            "canonicalize_conv_autopad",
            "eliminate_unused_input",
            "eliminate_unused_outputs",
        ]
    )
    def onnx_simplifier(graph: OnnxGraph):
        """Simplify onnx graph"""
        model_sim, succeed = onnxsim.simplify(graph.model)
        if succeed:
            # FIXME: onnxsim ignores functions
            model_sim.functions.extend(graph.functions.values())
            return OnnxGraph(model_sim)
        return graph
