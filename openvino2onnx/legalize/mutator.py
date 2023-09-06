"""
Copyright Wenyi Tang 2023

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

"""

import networkx as nx


class SingleNodeMutator:
    """Mutate a single node in the graph."""

    def __init__(self, pattern):
        self._pattern = pattern

    @property
    def pattern(self):
        return self._pattern

    def __call__(self, graph: nx.MultiDiGraph):
        for i in list(graph):
            if i not in graph:
                # node may be removed during transformation
                continue
            if graph.nodes[i]["type"] == self._pattern:
                if "_visited" in graph.nodes[i]:
                    continue
                self.trans(graph, i)
                if i in graph:
                    graph.nodes[i]["_visited"] = True
