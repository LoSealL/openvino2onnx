"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

import os
import tempfile
from contextlib import suppress
from pathlib import Path

import pytest
from omz_tools.omz_converter import converter
from omz_tools.omz_downloader import download


def prepare_models():
    """Make a generator to download and convert omz models."""
    with open(Path(__file__).parent / "models/model.list", encoding="utf-8") as lst:
        model_list = lst.readlines()

    for model in model_list:
        with tempfile.TemporaryDirectory() as cache_dir:
            args = [
                "--name",
                model.strip("\n"),
                "--precisions",
                "FP32",
                "-o",
                str(cache_dir),
                "-j",
                str(os.cpu_count()),
            ]
            try:
                download(args)
                with suppress(Exception):
                    converter(args + ["-d", str(cache_dir)])
                    yield from Path(cache_dir).rglob("*.xml")
            except SystemExit:
                # download error
                continue


MAX_MODEL = 1000


def pytest_generate_tests(metafunc: pytest.Metafunc):
    """Generate parametrized arguments to all tests with arg 'model'.
    `model` is acquired from model_zoo.
    """

    model_generator = prepare_models()
    models = [lambda: next(model_generator) for _ in range(MAX_MODEL)]
    if "model_gen" in metafunc.fixturenames:
        metafunc.parametrize("model_gen", models)
