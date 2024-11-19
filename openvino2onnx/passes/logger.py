"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

import logging
import os
from logging import Formatter, StreamHandler, addLevelName, getLevelName
from typing import Literal, Union, overload


def _default_level_from_env():
    match os.environ.get("OPENVINO2ONNX_LOG_LEVEL"):
        case "LOG_TRACE" | "0":
            return 1
        case "LOG_DEBUG" | "1":
            return logging.DEBUG
        case "LOG_INFO" | "2":
            return logging.INFO
        case "LOG_WARNING" | "3":
            return logging.WARNING
        case "LOG_ERROR" | "4":
            return logging.ERROR
    return logging.INFO


def trace(msg: str):
    """Logging trace message"""
    _LOG.log(1, msg)


def debug(msg: str):
    """Logging debug message"""
    _LOG.debug(msg)


def info(msg: str):
    """Logging informative message"""
    _LOG.info(msg)


def warning(msg: str):
    """Logging warning message"""
    _LOG.warning(msg)


def error(msg: str):
    """Logging error message"""
    _LOG.error(msg)


def fatal(msg: str):
    """Logging critical(fatal) message"""
    _LOG.critical(msg)


def set_level(level: str):
    """Filter log messages by different level."""
    _HDL.setLevel(level.upper())


def is_enabled_for(level: Union[int, str]) -> bool:
    """Whether current log level is active."""
    if isinstance(level, str):
        from logging import _nameToLevel  # pylint:disable=import-outside-toplevel

        return _LOG.isEnabledFor(_nameToLevel[level])
    return _LOG.isEnabledFor(level)


@overload
def get_level() -> str: ...  # noqa: E704


@overload
def get_level(use_string: Literal[True]) -> str: ...  # noqa: E704


@overload
def get_level(use_string: Literal[False]) -> int: ...  # noqa: E704


def get_level(use_string: bool = True) -> Union[int, str]:
    """Get current logging level."""
    if use_string:
        return getLevelName(_HDL.level)
    return _HDL.level


def drop():
    """Drop the logger handlers and flush pending events.
    Logging after `drop` will do nothing.
    """
    for hdl in list(_LOG.handlers):
        hdl.flush()
        hdl.close()
        _LOG.removeHandler(hdl)


def nest(nested: str):
    """Create a nested logger with nested name."""
    log = logging.getLogger(f"O2O.{nested}")
    log.setLevel(_LOG.level)
    return log


addLevelName(1, "TRACE")
_LOG = logging.getLogger("O2O")
_LOG.setLevel("TRACE")
_HDL = StreamHandler()
_HDL.setLevel(_default_level_from_env())
# use colorlog to colorize different level messages
try:
    import colorlog  # pylint:disable=import-outside-toplevel

    _HDL.setFormatter(
        colorlog.ColoredFormatter(
            "[%(asctime)s]%(log_color)s[%(levelname)s] %(message)s"
        )
    )
except ImportError:
    _HDL.setFormatter(Formatter("[%(asctime)s][%(levelname)s] %(message)s"))
_LOG.addHandler(_HDL)
