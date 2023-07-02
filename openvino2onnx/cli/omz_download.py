"""
Copyright Wenyi Tang 2023

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

"""

from pathlib import Path
from typing import Tuple

from openvino.model_zoo.omz_downloader import download

PROC = "omz://"


def parse_omz_link(link: str) -> Tuple[str, str]:
    """Parse a link text to model name and precision.

    A valid link should be one of:
    1. omz://gaze-estimation-adas-0002/FP16
    2. omz://gaze-estimation-adas-0002
    """
    if not link.startswith("omz://"):
        raise ValueError(f"Invalid OMZ link: {link}")
    proc_size = len(PROC)
    links = link[proc_size:].split("/", 2)
    match len(links):
        case 1:
            return links[0], "FP32"
        case 2:
            return links[0], links[1]
    raise ValueError(f"Invalid OMZ link: {link}")


def download_from_omz_link(omz_link: str, out_dir: str):
    """Download a model from openvino model zoo (OMZ)"""
    name, prec = parse_omz_link(omz_link)
    if not Path(out_dir).exists():
        Path(out_dir).mkdir(parents=True)
    download([f"--name={name}", f"--precisions={prec}", f"-o={out_dir}"])
    return next(Path(out_dir).glob(f"**/{name}/{prec}/**/*.xml"))
