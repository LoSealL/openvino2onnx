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


def subgraph_successor(graph: nx.DiGraph, subgraph: nx.DiGraph):
    """`subgraph` is a subgraph from `graph`, this function gets all successor
    nodes in the `graph` that connects the `subgraph`.
    """

    h = nx.induced_subgraph(graph, subgraph)
    if h.nodes != subgraph.nodes:
        raise ValueError(f"subgraph {subgraph} is not from {graph}!")
    succ = []
    for n in subgraph.nodes:
        if graph.out_degree(n) > subgraph.out_degree(n):
            for _, v in graph.out_edges(n):
                if v not in subgraph:
                    succ.append(v)
    return succ
