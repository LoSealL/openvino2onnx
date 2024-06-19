"""
Copyright Wenyi Tang 2023

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

"""

import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
import onnx
from openvino import runtime

from openvino2onnx import builder, ir11
from openvino2onnx.legalize import legalize


def build_model(model_path: Path, strict: bool = False):
    """Build onnx model from OpenVINO IR and compare the runtime outputs.

    Args:
        model_path (Path): path to xml file, the bin file must be at the same location.
        strict (bool, optional): If true, the output has to be bit matched.
            Defaults to False.
    """
    try:
        g = ir11.ir_to_graph(model_path)
        legalize(g)
        model = builder.build(g)
    except Exception:
        print(f"{model_path} convert error")
        raise

    onnx.checker.check_model(model)

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx.save(model, f"{tmpdir}/test.onnx")
        test = runtime.compile_model(f"{tmpdir}/test.onnx")
    ref = runtime.compile_model(model_path)

    inputs = [np.random.normal(0, 1, i.shape).astype("float32") for i in test.inputs]
    out_ref = ref(inputs)
    out_test = test(inputs)
    name_not_match = False
    errors = {}
    for k in out_ref:
        if not any(i in out_test for i in k.names):
            name_not_match = True
            continue
        for name in k.names:
            if name in out_test:
                if not np.allclose(out_ref[name], out_test[name]):
                    errors[name] = np.abs(out_ref[name] - out_test[name])
    if name_not_match and len(out_ref) == len(out_test):
        for i, (x, y) in enumerate(zip(out_ref.values(), out_test.values())):
            if not np.allclose(x, y):
                errors[i] = np.abs(x - y)
        name_not_match = False
    assert not name_not_match
    if strict:
        assert len(errors) == 0, f"{model_path} not bitmatch"
    else:
        for err in errors.values():
            assert err.mean() <= 1e-4, f"{model_path} precision error"


def test_build_models(model_gen: Callable[[], Path]):
    """Test a model from OMZ."""
    try:
        model = model_gen()
        build_model(model)
    except StopIteration:
        return


def test_build_models_strict(model_gen: Callable[[], Path]):
    """Test a model from OMZ with strict criteiria."""
    try:
        model = model_gen()
        build_model(model, strict=True)
    except StopIteration:
        return
