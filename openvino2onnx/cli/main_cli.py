"""
Copyright Wenyi Tang 2023

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

"""

import tempfile
from argparse import ArgumentParser
from pathlib import Path

import onnx

from openvino2onnx import build, ir_to_graph
from openvino2onnx.legalize import legalize


def parse_args():
    """Parse command line"""
    parser = ArgumentParser("openvino2onnx")
    parser.add_argument("model_url", help="an omz link or a local url to model file")
    parser.add_argument("--model-bin", "-b", help="specify a binary file of the model")
    parser.add_argument("--output", "-o", help="specify a output onnx file name")
    parser.add_argument(
        "--opset-version",
        "--opset",
        type=int,
        choices=(9, 10, 11, 12, 13),
        help="specify a version number of onnx opset",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_url = args.model_url
    if model_url.startswith("omz://"):
        # pylint: disable=import-outside-toplevel
        from openvino2onnx.cli.omz_download import download_from_omz_link

        with tempfile.TemporaryDirectory() as tmpdir:
            model_url = download_from_omz_link(model_url, tmpdir)
            graph = ir_to_graph(model_url)
    else:
        graph = ir_to_graph(model_url, args.model_bin)
    graph = legalize(graph)
    model = build(graph)
    if args.output is None:
        out_file = Path(model_url).stem + ".onnx"
    else:
        out_file = Path(args.output).with_suffix(".onnx")
    onnx.save(model, out_file)
    print(f"onnx file saved at {out_file}")
