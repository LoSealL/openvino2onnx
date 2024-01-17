"""
Copyright Wenyi Tang 2023

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

"""
# pylint: disable=W0718

import warnings
from typing import Iterator

from .mutator import SingleNodeMutator


class Compose:
    """Compose a list of graph mutators"""

    def __init__(self, mutators: Iterator[SingleNodeMutator]):
        self._fn = list(filter(callable, mutators))

    def __call__(self, graph):
        error_msg = []
        for fn in self._fn:
            try:
                fn(graph)
            except Exception as ex:  # noqa: E722
                warnings.warn(f"{type(fn)} throws an exception")
                error_msg.append(str(ex))
        if error_msg:
            raise RuntimeError("\n\n".join(error_msg))
        return graph

    def register(self, func):
        """Register new mutator."""
        self._fn.append(func())
        return func


legalize = Compose([])
