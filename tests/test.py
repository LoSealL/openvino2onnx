"""
Copyright Wenyi Tang 2023

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from openvino import runtime

from openvino2onnx import builder, ir11
from openvino2onnx.legalize import legalize


class TestOpenvinoConvert(unittest.TestCase):
    def build_model(self, model_path: str, strict: bool = False):
        g = ir11.ir_to_graph(model_path)
        legalize(g)
        model = builder.build(g)

        onnx.checker.check_model(model)

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx.save(model, f"{tmpdir}/test.onnx")
            test = runtime.compile_model(f"{tmpdir}/test.onnx")
        ref = runtime.compile_model(model_path)

        inputs = [
            np.random.normal(0, 1, i.shape).astype("float32") for i in test.inputs
        ]
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
        self.assertFalse(name_not_match)
        if strict:
            self.assertEqual(len(errors), 0)
        else:
            for err in errors.values():
                self.assertLessEqual(err.mean(), 1e-4)

    def test_build_models(self):
        cwd = Path(__file__).parent
        for model in cwd.rglob("*.xml"):
            with self.subTest(model=model):
                self.build_model(model)

    def test_build_models_strict(self):
        cwd = Path(__file__).parent
        for model in cwd.rglob("*.xml"):
            with self.subTest(model=model):
                self.build_model(model, strict=True)


if __name__ == "__main__":
    unittest.main()
