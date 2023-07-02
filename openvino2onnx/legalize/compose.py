"""
Copyright Wenyi Tang 2023

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

"""

import warnings
from typing import Iterator

from .mutator import SingleNodeMutator


class Compose:
    """Compose a list of graph mutators"""

    def __init__(self, mutators: Iterator[SingleNodeMutator]):
        self._fn = list(filter(callable, mutators))

    def __call__(self, graph):
        for fn in self._fn:
            try:
                fn(graph)
            except:  # noqa: E722
                warnings.warn(f"{type(fn)} throws an exception")
                raise
        return graph

    def register(self, func):
        """Register new mutator."""
        self._fn.append(func())
        return func


legalize = Compose([])
