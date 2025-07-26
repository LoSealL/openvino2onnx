# OPENVINO2ONNX
A simple tool to convert your IR XML to ONNX file.

[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)

Supported OpenVINO IR Version

- IRv10: openvino>=2020,<2023
- IRv11: openvino>=2023

## Usage

1. Install from PyPI
```shell
pip install openvino2onnx
```

2. Convert IR using CLI
```shell
openvino2onnx model.xml
```

```
usage: openvino2onnx input_model.xml [output_model.onnx]

openvino2onnx command-line api

options:
  -h, --help            show this help message and exit
  -a [ACTIVATE ...], --activate [ACTIVATE ...]
                        select passes to be activated, activate L1, L2 and L3 passes if not set.
  -r [REMOVE ...], --remove [REMOVE ...]
                        specify passes to be removed from activated passes.
  -n, --no-passes       do not run any optimizing passes, just convert the model
  --print [PRINT]       print the name of all optimizing passes
  --format {protobuf,textproto,json,onnxtxt}
                        onnx file format
  -s, --infer-shapes    infer model shapes
  -c CONFIG_FILE, --config-file CONFIG_FILE
                        specify a json-format config file for passes
  -u, --uncheck         no checking output model
  --check               check optimized model with random inputs
  --checker-backend {onnx,openvino,onnxruntime}
                        backend for accuracy checking, defaults to openvino
  -v OPSET_VERSION, --opset-version OPSET_VERSION
                        target opset version, defaults to 19
  -vv [{DEBUG,INFO,WARNING,ERROR,CRITICAL}], --log-level [{DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                        specify the level of log messages to be printed, defaults to INFO
```

To print pass information:

```
openvino2onnx --print all
openvino2onnx --print fuse_swish
openvino2onnx --print l1
```
