"""Unit tests for Kemeny constant calculations."""

import networkx as nx
from dss.analytics.kemeny import kemeny_constant, interactive_kemeny


def test_kemeny_constant_cycle():
    G = nx.cycle_graph(4)
    K = kemeny_constant(G)
    # Kemeny constant should be positive and finite
    assert K > 0
    assert K < 100


def test_interactive_kemeny_removal():
    G = nx.path_graph(4)
    result = interactive_kemeny(G, [0, 1])
    # History length should match number of removals
    assert len(result.history) == 2
    # After removing all but one node, Kemeny may be undefined (NaN) or zero
    # We just ensure no error occurred and K is a float
    assert isinstance(result.kemeny, float)