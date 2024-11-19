"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

"""

# pylint: disable=missing-function-docstring

import argparse
import json
from pathlib import Path

from . import OPENVINO2ONNX_OPSET, PassManager, convert_graph
from .passes.logger import set_level

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
        help="select passes to be activated, activate all passes if not set.",
    )
    parser.add_argument(
        "-r",
        "--remove",
        nargs="*",
        help="specify passes to be removed from activated passes.",
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
        passes=args.activate,
        exclude=args.remove,
        onnx_format=args.format,
        configs=configs,
        target_opset=args.opset_version,
    )
    graph.save(
        output_model,
        format=args.format,
        infer_shapes=args.infer_shapes,
        check=args.uncheck,
    )
    print(f"model saved to {output_model}")


if __name__ == "__main__":
    main()
