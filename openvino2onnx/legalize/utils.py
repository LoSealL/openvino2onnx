"""
Copyright Wenyi Tang 2023

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

"""

import networkx as nx


def get_node_on_edge(graph: nx.DiGraph, node, port):
    """Get a node that connected to the specified port of `node`."""
    for u, _, data in graph.in_edges(node, data=True):
        if data["dst"] == port:
            return u
    for _, v, data in graph.out_edges(node, data=True):
        if data["src"] == port:
            return v
    raise RuntimeError(f"No node on {graph.nodes[node]['name']}:{port}")
