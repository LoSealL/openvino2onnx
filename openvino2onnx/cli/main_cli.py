"""
Copyright Wenyi Tang 2023

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

"""

import tempfile
from argparse import ArgumentParser
from pathlib import Path

import onnx

from openvino2onnx import transform


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
        choices=set(range(11, 20)),
        default=17,
        help="specify a version number of onnx opset",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="a flag to forcely convert float16 to float32",
    )
    return parser.parse_args()


def main():
    """Entry"""
    args = parse_args()
    model_url = args.model_url
    if model_url.startswith("omz://"):
        # pylint: disable=import-outside-toplevel
        from openvino2onnx.cli.omz_download import download_from_omz_link

        with tempfile.TemporaryDirectory() as tmpdir:
            model_url = download_from_omz_link(model_url, tmpdir)
            model = transform(model_url, None, args.fp32, args.opset_version)
    else:
        model = transform(model_url, args.model_bin, args.fp32, args.opset_version)
    if args.output is None:
        out_file = Path(model_url).stem + ".onnx"
    else:
        out_file = Path(args.output).with_suffix(".onnx")
    onnx.save_model(model, out_file)
    print(f"[I] onnx file saved at {out_file}")
