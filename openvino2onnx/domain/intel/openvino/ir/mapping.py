"""
Copyright (C) 2024 The OPENVINO2ONNX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from onnx.mapping import TENSOR_TYPE_MAP

ETYPE2DTYPE = {
    "f64": "float64",
    "f32": "float32",
    "f16": "float16",
    "bf16": "bfloat16",
    "i64": "int64",
    "i32": "int32",
    "i16": "int16",
    "i8": "int8",
    "u64": "uint64",
    "u32": "uint32",
    "u16": "uint16",
    "u8": "uint8",
    "boolean": "bool",
}
"""Mapping ov element-type to numpy dtype."""

PREC2DTYPE = {
    "FP64": "float64",
    "FP32": "float32",
    "FP16": "float16",
    "BF16": "bfloat16",
    "I64": "int64",
    "I32": "int32",
    "I16": "int16",
    "I8": "int8",
    "U64": "uint64",
    "U32": "uint32",
    "U16": "uint16",
    "U8": "uint8",
    "BOOL": "bool",
}
"""Mapping ov precision to numpy dtype."""

DTYPE2TENSORTYPE = {v.np_dtype: k for k, v in TENSOR_TYPE_MAP.items() if k < 16}
"""Mapping numpy dtype to tensor type id."""

DTYPE2PREC = {v: k for k, v in PREC2DTYPE.items()}
"""Reverse mapping numpy dtype to precision."""
