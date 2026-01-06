"""Unit tests for centrality computation shapes."""

import networkx as nx
from dss.analytics.centrality import compute_centralities


def test_compute_centralities_shape():
    G = nx.path_graph(5)
    df = compute_centralities(G)
    # Should have one row per node
    assert df.shape[0] == 5
    # Should have at least the expected columns
    expected_cols = {"degree", "katz", "eigenvector", "betweenness", "closeness", "pagerank"}
    assert expected_cols.issubset(df.columns)