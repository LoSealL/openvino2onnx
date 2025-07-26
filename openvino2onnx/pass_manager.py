"""
Copyright (C) 2024-2025 The OPENVINO2ONNX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import traceback
from functools import partial
from itertools import chain
from typing import Any, Dict, List, Mapping, Optional, Sequence

import networkx as nx
from tabulate import tabulate
from termcolor import colored

from .graph import OnnxGraph
from .logger import debug, error, warning
from .passes import L1, L2, L3, PASSES
from .traits import RewriterInterface


class PassManager:
    """Ordered optimization pass list.

    Args:
        include (List[str | RewriterInterface], Optional): a list of pattern to
            select passes. Defaults to select L1, L2 and L3 passes.
        exclude (List[str], Optional): a list of pattern to deselect passes.
            Defaults to None.
        configs (Dict[str, Any], Optional): a dictionary of pass configurations.
            Defaults to None.
    """

    def __init__(
        self,
        include: Optional[Sequence[str | RewriterInterface]] = None,
        exclude: Optional[Sequence[str]] = None,
        configs: Optional[Dict[str, Any]] = None,
    ) -> None:
        passes: List[str | RewriterInterface]
        if include is None:
            passes = [i for i in chain(L1, L2, L3)]
        else:
            passes = [i for i in include]
        if exclude:
            passes = list(filter(lambda i: i not in exclude, passes))
        activated: List[RewriterInterface] = []
        for i in passes:
            if isinstance(i, str):
                if i in PASSES:
                    activated.append(PASSES[i])
                else:
                    warning(f"{i} is not registered as a pass, ignore it.")
            else:
                activated.append(i)
        self.activated: List[RewriterInterface] = activated
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
            candidates = [i for i in self.activated if i.__NAME__ == key]
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
                self.activated[pos].__NAME__ = key
                self.activated[pos].__DEPS__ = func.__DEPS__
                self.activated[pos].__PATCHES__ = func.__PATCHES__

    def _expand(self, nodes, priv_member):
        root: nx.DiGraph = nx.DiGraph()  # type: ignore
        leaves = [f"{i}:{n}" for i, n in enumerate(nodes)]
        root.add_nodes_from(leaves)
        # a shallow graph to check cyclic dependencies
        shallow: nx.DiGraph = nx.DiGraph()  # type: ignore
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
        root = self._expand(deps, "__DEPS__")
        for i in nx.traversal.dfs_postorder_nodes(root):
            yield i.split(":")[1]

    def _expand_patches(self, nodes):
        root = self._expand(nodes, "__PATCHES__")
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
                for deps in self._expand_deps(opt.__DEPS__):
                    debug(f"Applying dependency pass: {deps}")
                    graph = PASSES[deps](graph) if deps in PASSES else graph
                debug(f"Applying pass: {opt.__NAME__}")
                graph = opt(graph)
                for patch in self._expand_patches(opt.__PATCHES__):
                    debug(f"Applying patch pass: {patch}")
                    graph = PASSES[patch](graph) if patch in PASSES else graph
            except Exception as ex:  # pylint: disable=broad-exception-caught
                error(f"{opt.__NAME__} failed: {ex}")
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
            [[i.__NAME__, i] for i in self.activated], ["PASS", "Func"], "grid"
        )


def print_pass_simple(pm: PassManager):
    """Print activated passes of a PassManager in a simple format."""

    msg = ""
    for n, i in enumerate(pm.activated):
        deps = []
        post = []
        # pylint: disable=protected-access
        for j in pm._expand_deps(i.__DEPS__):
            if j in PASSES:
                deps.append(j)
        for j in pm._expand_patches(i.__PATCHES__):
            if j in PASSES:
                post.append(j)
        msg += f"\n{n:<2} "
        if deps:
            msg += "[" + colored(",".join(deps), "yellow") + "] "
        msg += i.__NAME__
        if post:
            msg += " [" + colored(",".join(post), "magenta") + "]"
    if msg:
        print(f"Activated passes ([deps] pass [patches]):{msg}")
