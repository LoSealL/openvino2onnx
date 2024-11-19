"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

from contextlib import suppress
from typing import List, Literal, Optional


def text_to_boolean(text: Literal["0", "1", "false", "true"]) -> bool:
    """Convert common text to bool

    Args:
        text (str): integer string or false/true literals
    """

    with suppress(ValueError):
        integer = int(text)
        return bool(integer)
    return text.lower() == "true"


def text_to_integers(text: Optional[str]) -> List[int]:
    """Convert comma-separated text to list of integers

    Args:
        text (str | None): comma-separated integers
    """

    if text is None:
        return None
    elif text.strip() == "":
        return []
    return list(map(int, text.split(",")))
