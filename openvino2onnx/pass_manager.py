"""
Copyright Wenyi Tang 2024-2025

:Author: Wenyi Tang
:Email: wenyitang@outlook.com
"""

import traceback
from abc import ABCMeta, abstractmethod
from functools import partial
from itertools import chain
from typing import Any, Dict, List, Mapping, Optional, Sequence

import networkx as nx
from tabulate import tabulate

from .graph import OnnxGraph
from .passes import L1, L2, L3, PASSES
from .passes.logger import debug, error, warning


class RewriterInterface(metaclass=ABCMeta):
    """Interface helper for rewriting ONNX graphs."""

    @abstractmethod
    def __call__(self, graph: OnnxGraph, *args, **kwargs) -> OnnxGraph:
        """Rewriter is a callable that takes :class:`OnnxGraph` and returns a
        modified graph.
        """

    @property
    @abstractmethod
    def __deps__(self) -> List[str]:
        """This property is a list of pass names that this pass depends on."""

    @__deps__.setter
    def __deps__(self, value: Sequence[str]):
        """This property can be changed."""

    @property
    @abstractmethod
    def __patches__(self) -> List[str]:
        """This property is a list of pass names that will be applied after
        this pass."""

    @__patches__.setter
    def __patches__(self, value: Sequence[str]):
        """This property can be changed."""

    @property
    @abstractmethod
    def __name__(self) -> str:
        """This property is the name of the pass."""

    @__name__.setter
    def __name__(self, value: str):
        """This property can be changed."""


class PassManager:
    """Ordered optimization pass list.

    Args:
        include (List[str], Optional): a list of pattern to select passes.
            Defaults to select all passes.
        exclude (List[str], Optional): a list of pattern to deselect passes.
            Defaults to None.
        configs (Dict[str, Any], Optional): a dictionary of pass configurations.
            Defaults to None.
    """

    def __init__(
        self,
        include: Optional[Sequence[str]] = None,
        exclude: Optional[Sequence[str]] = None,
        configs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if include is None:
            passes = [i for i in chain(L1, L2, L3)]
        else:
            passes = [i for i in include]
        if exclude:
            passes = list(filter(lambda i: i not in exclude, passes))
        self.activated: List[RewriterInterface] = [  # type: ignore
            PASSES[i] for i in passes if i in PASSES  # type: ignore
        ]
        if configs:
            self._assign_config_to_pass(configs)

    def _assign_config_to_pass(self, configs: Dict[str, Any]):
        for key, config in configs.items():
            index = -1
            if ":" in key:
                key, index_str = key.split(":", 2)
                index = int(index_str)
            if not isinstance(config, Mapping):
                warning(f"config {key}:{index} must be a dict, but got {type(config)}")
                continue
            candidates = [i for i in self.activated if i.__name__ == key]
            if index >= 0 and index >= len(candidates):
                warning(
                    f"config {key}:{index} exceeds the boundary. "
                    f"Number of {key} is {len(candidates)}"
                )
                continue
            if index >= 0:
                candidates = [candidates[index]]
            for func in candidates:
                pos = self.activated.index(func)
                self.activated[pos] = partial(func, **config)  # type: ignore
                self.activated[pos].__name__ = key
                self.activated[pos].__deps__ = func.__deps__
                self.activated[pos].__patches__ = func.__patches__

    def _expand(self, nodes, priv_member):
        root = nx.DiGraph()
        leaves = [f"{i}:{n}" for i, n in enumerate(nodes)]
        root.add_nodes_from(leaves)
        shallow = nx.DiGraph()  # a shallow graph to check cyclic dependencies
        shallow.add_nodes_from(nodes)
        ind = len(leaves)
        while leaves:
            leaf = leaves.pop(0)
            leaf_pass = leaf.split(":")[1]
            children = getattr(PASSES.get(leaf_pass), priv_member, [])
            shallow.add_edges_from([(leaf_pass, child) for child in children])
            try:
                cycles = nx.find_cycle(shallow, leaf_pass)
            except nx.NetworkXNoCycle:
                children = [f"{ind + i}:{c}" for i, c in enumerate(children)]
                ind += len(children)
                root.add_edges_from([(leaf, child) for child in children])
                leaves.extend(children)
            else:
                error(f"Cyclic dependencies found!: {cycles}")
                raise RuntimeError("Cyclic dependencies found!")
        return root

    def _expand_deps(self, deps):
        root = self._expand(deps, "__deps__")
        for i in nx.traversal.dfs_postorder_nodes(root):
            yield i.split(":")[1]

    def _expand_patches(self, nodes):
        root = self._expand(nodes, "__patches__")
        for i in nx.traversal.dfs_preorder_nodes(root):
            yield i.split(":")[1]

    def optimize(self, graph: OnnxGraph, strict: bool = False) -> OnnxGraph:
        """Invoke passes on the input graph.

        Args:
            graph (OnnxGraph): See :class:`OnnxGraph`.
            strict (bool): Break if any pass fails.
        """
        for opt in self.activated:
            try:
                for deps in self._expand_deps(opt.__deps__):
                    graph = PASSES[deps](graph) if deps in PASSES else graph
                graph = opt(graph)
                for patch in self._expand_patches(opt.__patches__):
                    graph = PASSES[patch](graph) if patch in PASSES else graph
            except Exception as ex:  # pylint: disable=broad-exception-caught
                error(f"{opt.__name__} failed: {ex}")
                debug("\n".join(traceback.format_exception(ex)))
                if strict:
                    raise
        return graph

    @classmethod
    def print_all(cls):
        """Print the name of all passes."""
        print(PASSES, flush=True)

    @classmethod
    def print_l1(cls):
        """Print the name of all L1 passes."""
        print(L1, flush=True)

    @classmethod
    def print_l2(cls):
        """Print the name of all L2 passes."""
        print(L2, flush=True)

    @classmethod
    def print_l3(cls):
        """Print the name of all L3 passes."""
        print(L3, flush=True)

    @classmethod
    def print(cls, names: str | List[str]):
        """Print a specific pass or a set of passes."""
        print(PASSES.child(names), flush=True)

    def __repr__(self) -> str:
        return tabulate(
            [[i.__name__, i] for i in self.activated], ["PASS", "Func"], "grid"
        )
