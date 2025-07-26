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

# pylint: disable=missing-function-docstring

import argparse
import json
from pathlib import Path

from . import OPENVINO2ONNX_OPSET, PassManager, convert_graph
from .checker import check_accuracy
from .logger import set_level

USAGE = "openvino2onnx input_model.onnx [output_model.onnx]"


def parse_args():
    parser = argparse.ArgumentParser(
        prog="openvino2onnx",
        usage=USAGE,
        description="openvino2onnx command-line api",
    )
    parser.add_argument(
        "-a",
        "--activate",
        nargs="*",
        help="select passes to be activated, activate L1, L2 and L3 passes if not set.",
    )
    parser.add_argument(
        "-r",
        "--remove",
        nargs="*",
        help="specify passes to be removed from activated passes.",
    )
    parser.add_argument(
        "-n",
        "--no-passes",
        action="store_true",
        help="do not run any optimizing passes, just convert the model",
    )
    parser.add_argument(
        "--print",
        nargs="?",
        const="all",
        help="print the name of all optimizing passes",
    )
    parser.add_argument(
        "--format",
        choices=("protobuf", "textproto", "json", "onnxtxt"),
        default=None,
        help="onnx file format",
    )
    parser.add_argument(
        "-s",
        "--infer-shapes",
        action="store_true",
        help="infer model shapes",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        help="specify a json-format config file for passes",
    )
    parser.add_argument(
        "-u",
        "--uncheck",
        action="store_false",
        help="no checking output model",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="check optimized model with random inputs",
    )
    parser.add_argument(
        "--checker-backend",
        choices=("onnx", "openvino", "onnxruntime"),
        default="onnxruntime",
        help="backend for accuracy checking, defaults to openvino",
    )
    parser.add_argument(
        "-v",
        "--opset-version",
        type=int,
        help=f"target opset version, defaults to {OPENVINO2ONNX_OPSET.version}",
    )
    parser.add_argument(
        "-vv",
        "--log-level",
        nargs="?",
        const="DEBUG",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="specify the level of log messages to be printed, defaults to INFO",
    )

    return parser.parse_known_args()


def main():
    args, argv = parse_args()
    if args.log_level:
        set_level(args.log_level)

    if args.print:
        match args.print:
            case "all":
                PassManager.print_all()
            case "l1":
                PassManager.print_l1()
            case "l2":
                PassManager.print_l2()
            case "l3":
                PassManager.print_l3()
            case _:
                PassManager.print(args.print)
        exit(0)
    if len(argv) == 1:
        input_model = Path(argv[0]).expanduser()
        output_model = Path(input_model.stem + "_o2o").expanduser()
    elif len(argv) == 2:
        input_model = Path(argv[0]).expanduser()
        output_model = Path(argv[1]).expanduser()
    else:
        print("Usage: " + USAGE)
        if len(argv) == 0:
            raise RuntimeError("missing input model")
        else:
            raise RuntimeError("unknown argument: " + ",".join(argv[2:]))

    if args.config_file:
        config_file = Path(args.config_file).expanduser()
        with open(config_file, encoding="utf-8") as file:
            configs = json.load(file)
    else:
        configs = None

    graph = convert_graph(
        model=input_model,
        passes=[] if args.no_passes else args.activate,
        exclude=args.remove,
        onnx_format=args.format,
        configs=configs,
        target_opset=args.opset_version,
    )
    output_model = graph.save(
        output_model,
        format=args.format,
        infer_shapes=args.infer_shapes,
        check=args.uncheck,
    )
    print(f"model saved to {output_model}")
    if args.check and isinstance(output_model, Path):
        error_maps = check_accuracy(
            input_model,
            output_model,
            backend=args.checker_backend,
        )
        for k, v in error_maps.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
