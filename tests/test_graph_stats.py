"""Unit tests for graph statistics functions."""

import networkx as nx
from dss.graph.stats import basic_statistics


def test_basic_statistics_path_graph():
    G = nx.path_graph(5)
    stats = basic_statistics(G)
    assert stats["N"] == 5
    assert stats["E"] == 4
    # Density of path graph with 5 nodes: 2*4/(5*4) = 0.4
    assert abs(stats["density"] - 0.4) < 1e-6
    assert stats["components"] == 1
    assert stats["component_sizes"] == [5]