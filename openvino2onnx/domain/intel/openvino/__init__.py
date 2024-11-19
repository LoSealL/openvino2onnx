"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from pathlib import Path

from openvino2onnx.passes.auto_load import auto_load

auto_load(Path(__file__).parent / "passes")
