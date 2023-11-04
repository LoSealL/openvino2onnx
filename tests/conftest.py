"""
INTEL CONFIDENTIAL

Copyright (C) 2023 Intel Corporation. All Rights Reserved.

The source code contained or described herein and all documents
related to the source code ("Material") are owned by Intel Corporation
or licensors. Title to the Material remains with Intel
Corporation or its licensors. The Material contains trade
secrets and proprietary and confidential information of Intel or its
licensors. The Material is protected by worldwide copyright
and trade secret laws and treaty provisions. No part of the Material may
be used, copied, reproduced, modified, published, uploaded, posted,
transmitted, distributed, or disclosed in any way without Intel's prior
express written permission.

No License under any patent, copyright, trade secret or other intellectual
property right is granted to or conferred upon you by disclosure or
delivery of the Materials, either expressly, by implication, inducement,
estoppel or otherwise. Any license under such intellectual property rights
must be express and approved by Intel in writing.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from contextlib import suppress
from pathlib import Path

import pytest
from openvino.model_zoo.omz_converter import (
    ModelOptimizerProperties,
    _concurrency,
    _reporting,
    convert,
    num_jobs_arg,
)
from openvino.model_zoo.omz_downloader import (
    Downloader,
    DownloaderArgumentParser,
    _common,
    _configuration,
    positive_int_arg,
)


def converter(argv):
    """Convert OMZ models without exiting itself."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--download_dir", type=Path, metavar="DIR")
    parser.add_argument("-o", "--output_dir", type=Path, metavar="DIR")
    parser.add_argument("--name", metavar="PAT[,PAT...]")
    parser.add_argument("--list", type=Path, metavar="FILE.LST")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--precisions", metavar="PREC[,PREC...]")
    parser.add_argument(
        "-p", "--python", type=Path, metavar="PYTHON", default=sys.executable
    )
    parser.add_argument("--mo", type=Path, metavar="MO.PY")
    parser.add_argument(
        "--add_mo_arg", dest="extra_mo_args", metavar="ARG", action="append"
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("-j", "--jobs", type=num_jobs_arg, default=1)

    args = parser.parse_args(argv)

    with _common.telemetry_session("Model Converter", "converter") as telemetry:
        args_count = sum([args.all, args.name is not None, args.list is not None])
        if args_count == 0:
            telemetry.send_event("md", "converter_selection_mode", None)
        else:
            for mode in ["all", "list", "name"]:
                if getattr(args, mode):
                    telemetry.send_event("md", "converter_selection_mode", mode)

        models = _configuration.load_models_from_args(parser, args, _common.MODEL_ROOT)

        if args.precisions is None:
            requested_precisions = _common.KNOWN_PRECISIONS
        else:
            requested_precisions = set(args.precisions.split(","))

        for model in models:
            precisions_to_send = (
                requested_precisions
                if args.precisions
                else requested_precisions & model.precisions
            )
            model_information = {
                "name": model.name,
                "framework": model.framework,
                "precisions": str(sorted(precisions_to_send)).replace(",", ";"),
            }
            telemetry.send_event("md", "converter_model", json.dumps(model_information))

        mo_path = args.mo

        if mo_path is None:
            mo_executable = shutil.which("mo")

            if mo_executable:
                mo_path = Path(mo_executable)
            else:
                raise OSError("Unable to locate Model Optimizer.")

        if mo_path is not None:
            mo_path = mo_path.resolve()
            mo_cmd_prefix = [str(args.python), "--", str(mo_path)]

            if str(mo_path).lower().endswith(".py"):
                mo_dir = mo_path.parent
            else:
                mo_package_path, stderr = _common.get_package_path(
                    args.python, "openvino.tools.mo"
                )
                mo_dir = mo_package_path

                if mo_package_path is None:
                    mo_package_path, stderr = _common.get_package_path(
                        args.python, "mo"
                    )
                    if mo_package_path is None:
                        raise OSError(
                            f"Unable to load Model Optimizer. Errors occurred: {stderr}"
                        )
                    mo_dir = mo_package_path.parent

        output_dir = args.download_dir if args.output_dir is None else args.output_dir

        reporter = _reporting.Reporter(_reporting.DirectOutputContext())
        mo_props = ModelOptimizerProperties(
            cmd_prefix=mo_cmd_prefix,
            extra_args=args.extra_mo_args or [],
            base_dir=mo_dir,
        )
        shared_convert_args = (output_dir, args, mo_props, requested_precisions)

        def convert_model(model, reporter):
            if model.model_stages:
                results = []
                for model_stage in model.model_stages:
                    results.append(convert(reporter, model_stage, *shared_convert_args))
                return sum(results) == len(model.model_stages)
            else:
                return convert(reporter, model, *shared_convert_args)

        if args.jobs == 1 or args.dry_run:
            results = [convert_model(model, reporter) for model in models]
        else:
            results = _concurrency.run_in_parallel(
                args.jobs,
                lambda context, model: convert_model(
                    model, _reporting.Reporter(context)
                ),
                models,
            )

        failed_models = [
            model.name for model, successful in zip(models, results) if not successful
        ]

        if failed_models:
            reporter.print("FAILED:")
            for failed_model_name in failed_models:
                reporter.print(failed_model_name)
            raise OSError()


def download(argv):
    """An OMZ downloader without exiting itself."""
    parser = DownloaderArgumentParser()
    parser.add_argument("--name", metavar="PAT[,PAT...]")
    parser.add_argument("--list", type=Path, metavar="FILE.LST")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--print_all", action="store_true")
    parser.add_argument("--precisions", metavar="PREC[,PREC...]")
    parser.add_argument(
        "-o", "--output_dir", type=Path, metavar="DIR", default=Path.cwd()
    )
    parser.add_argument("--cache_dir", type=Path, metavar="DIR")
    parser.add_argument("--num_attempts", type=positive_int_arg, metavar="N", default=1)
    parser.add_argument("--progress_format", choices=("text", "json"), default="text")
    # unlike Model Converter, -jauto is not supported here, because CPU count has no
    # relation to the optimal number of concurrent downloads
    parser.add_argument("-j", "--jobs", type=positive_int_arg, metavar="N", default=1)

    args = parser.parse_args(argv)

    reporter = Downloader.make_reporter(args.progress_format)

    with _common.telemetry_session("Model Downloader", "downloader") as telemetry:
        args_count = sum([args.all, args.name is not None, args.list is not None])
        if args_count == 0:
            telemetry.send_event("md", "downloader_selection_mode", None)
        else:
            for mode in ["all", "list", "name"]:
                if getattr(args, mode):
                    telemetry.send_event("md", "downloader_selection_mode", mode)

        models = _configuration.load_models_from_args(parser, args, _common.MODEL_ROOT)
        failed_models = set()

        if args.precisions is None:
            requested_precisions = _common.KNOWN_PRECISIONS
        else:
            requested_precisions = set(args.precisions.split(","))

        for model in models:
            precisions_to_send = (
                requested_precisions
                if args.precisions
                else requested_precisions & model.precisions
            )
            model_information = {
                "name": model.name,
                "framework": model.framework,
                "precisions": str(sorted(precisions_to_send)).replace(",", ";"),
            }
            telemetry.send_event(
                "md", "downloader_model", json.dumps(model_information)
            )

        downloader = Downloader(
            requested_precisions, args.output_dir, args.cache_dir, args.num_attempts
        )

        failed_models = downloader.bulk_download_model(
            models, reporter, args.jobs, args.progress_format
        )

        if failed_models:
            reporter.print("FAILED:")
            for failed_model_name in failed_models:
                reporter.print(failed_model_name)
                telemetry.send_event(
                    "md", "downloader_failed_models", failed_model_name
                )
            raise OSError()


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
            with suppress(OSError):
                download(args)
                with suppress(Exception):
                    converter(args + ["-d", str(cache_dir)])
                    yield from Path(cache_dir).rglob("*.xml")


MAX_MODEL = 1000


def pytest_generate_tests(metafunc: pytest.Metafunc):
    """Generate parametrized arguments to all tests with arg 'model'.
    `model` is acquired from model_zoo.
    """

    model_generator = prepare_models()
    models = [lambda: next(model_generator) for _ in range(MAX_MODEL)]
    if "model_gen" in metafunc.fixturenames:
        metafunc.parametrize("model_gen", models)
