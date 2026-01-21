# File: src/dss/analytics/kemeny.py
"""Compute the Kemeny constant of a graph and evaluate its sensitivity."""

from dataclasses import dataclass
from typing import List, Any, Optional, Tuple

import numpy as np
import networkx as nx

from ..types import KemenyResult
from ..logging_config import get_logger

logger = get_logger(__name__)

Edge = Tuple[Any, Any]


@dataclass(frozen=True)
class EdgeKemenyResult:
    """Result object for edge removals (keeps node-based KemenyResult intact)."""

    kemeny: float
    removed_edges: List[Edge]
    history: List[float]


def _transition_matrix(G: nx.Graph) -> np.ndarray:
    """Build the transition matrix of a simple random walk on the graph."""
    nodes = list(G.nodes())
    n = len(nodes)
    idx = {node: i for i, node in enumerate(nodes)}
    P = np.zeros((n, n), dtype=float)
    for u in nodes:
        i = idx[u]
        deg = G.degree(u)
        if deg > 0:
            for v in G.neighbors(u):
                j = idx[v]
                P[i, j] = 1.0 / deg
    return P


def _stationary_distribution(G: nx.Graph, P: np.ndarray) -> np.ndarray:
    """Compute a stationary distribution for the transition matrix.

    For connected undirected graphs the stationary distribution is
    proportional to node degrees; for directed or disconnected graphs a
    power iteration approach is used.
    """
    n = G.number_of_nodes()
    if n == 0:
        return np.array([], dtype=float)

    if (not G.is_directed()) and nx.is_connected(G):
        degrees = np.array([deg for _, deg in G.degree()], dtype=float)
        total = degrees.sum()
        if total == 0:
            return np.full(n, 1.0 / n)
        pi = degrees / total
        return pi

    # Use power iteration: start with uniform distribution
    pi = np.full(n, 1.0 / n, dtype=float)
    for _ in range(100):
        pi_next = pi @ P
        # Normalise
        s = pi_next.sum()
        if s == 0:
            pi_next = np.full(n, 1.0 / n, dtype=float)
        else:
            pi_next = pi_next / s
        if np.allclose(pi, pi_next, atol=1e-10):
            break
        pi = pi_next
    return pi


def _is_connected_or_ergodic_ready(G: nx.Graph) -> bool:
    """Connectivity check used to decide if Kemeny is well-defined.

    For undirected graphs: connected.
    For directed graphs: weakly connected (still not enough for ergodicity, but aligns with current code style).
    """
    if G.number_of_nodes() == 0:
        return False
    if G.is_directed():
        return nx.is_weakly_connected(G)
    return nx.is_connected(G)


def _largest_component_subgraph(G: nx.Graph) -> nx.Graph:
    """Return a copy of the largest component subgraph."""
    if G.number_of_nodes() == 0:
        return G.copy()

    if G.is_directed():
        comps = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
        if not comps:
            return G.copy()
        return G.subgraph(comps[0]).copy()

    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    if not comps:
        return G.copy()
    return G.subgraph(comps[0]).copy()


def kemeny_constant(G: nx.Graph) -> float:
    """Compute the Kemeny constant of the graph.

    The Kemeny constant is defined as `K = trace(Z)`, where
    `Z = (I - P + 1π)^{-1}` is the fundamental matrix of the Markov chain
    defined by the transition matrix `P` and stationary distribution `π`.
    For disconnected graphs the constant is computed on the largest connected
    component to ensure the chain is ergodic.

    Parameters
    ----------
    G: networkx.Graph
        Graph on which to compute the Kemeny constant.

    Returns
    -------
    float
        The Kemeny constant of the graph.
    """
    if G.number_of_nodes() == 0:
        return 0.0

    # Use largest component if disconnected
    if not _is_connected_or_ergodic_ready(G):
        largest = _largest_component_subgraph(G)
        return kemeny_constant(largest)

    P = _transition_matrix(G)
    pi = _stationary_distribution(G, P)

    n = G.number_of_nodes()
    I = np.eye(n)
    one = np.ones((n, 1))

    try:
        Z_inv = I - P + one @ pi.reshape(1, -1)
        Z = np.linalg.inv(Z_inv)
        K = np.trace(Z)
        return float(np.real(K))
    except Exception as e:
        logger.warning(f"Failed to compute Kemeny constant: {e}")
        return float("nan")


def kemeny_after_removals(
    G: nx.Graph,
    removed_nodes: List[Any],
    recompute_on_largest: bool = True,
) -> float:
    """Compute Kemeny constant after removing specified nodes."""
    H = G.copy()
    H.remove_nodes_from(removed_nodes)

    if H.number_of_nodes() == 0:
        return float("nan")

    if _is_connected_or_ergodic_ready(H):
        return kemeny_constant(H)

    if recompute_on_largest:
        largest = _largest_component_subgraph(H)
        return kemeny_constant(largest)

    return float("nan")


def interactive_kemeny(
    G: nx.Graph,
    selected_nodes: List[Any],
    recompute_on_largest: bool = True,
) -> KemenyResult:
    """Return a KemenyResult tracking node removals and Kemeny values."""
    history: List[float] = []
    removed_so_far: List[Any] = []

    for node in selected_nodes:
        removed_so_far.append(node)
        k_val = kemeny_after_removals(G, removed_so_far, recompute_on_largest)
        history.append(k_val)

    current_k = kemeny_after_removals(G, removed_so_far, recompute_on_largest)
    return KemenyResult(kemeny=current_k, removed_nodes=removed_so_far, history=history)

def kemeny_after_edge_removals(
    G: nx.Graph,
    removed_edges: List[Edge],
    recompute_on_largest: bool = True,
) -> float:
    """Compute Kemeny constant after removing specified edges.

    Parameters
    ----------
    G: networkx.Graph
        Original graph.
    removed_edges: list of (u, v)
        Edges to remove before computing the Kemeny constant.
    recompute_on_largest: bool, optional
        If True, compute the constant only on the largest component when disconnected.
        Otherwise return NaN when disconnected.

    Returns
    -------
    float
        The Kemeny constant of the pruned graph or NaN if undefined.
    """
    H = G.copy()

    for u, v in removed_edges:
        if H.has_edge(u, v):
            H.remove_edge(u, v)
        elif (not H.is_directed()) and H.has_edge(v, u):
            # Undirected safety if input edge was flipped
            H.remove_edge(v, u)

    if H.number_of_nodes() == 0:
        return float("nan")

    # If graph has nodes but no edges, transition matrix still exists,
    # but random-walk structure becomes degenerate; keep behavior consistent:
    if not _is_connected_or_ergodic_ready(H):
        if recompute_on_largest:
            largest = _largest_component_subgraph(H)
            return kemeny_constant(largest)
        return float("nan")

    return kemeny_constant(H)


def interactive_kemeny_edges(
    G: nx.Graph,
    selected_edges: List[Edge],
    recompute_on_largest: bool = True,
) -> EdgeKemenyResult:
    """Return an EdgeKemenyResult tracking edge removals and Kemeny values.

    selected_edges are applied sequentially in the given order.
    history contains the Kemeny value after each step.
    """
    history: List[float] = []
    removed_so_far: List[Edge] = []

    if not recompute_on_largest and nx.is_connected(G):
        recompute_on_largest = True
    
    for e in selected_edges:
        removed_so_far.append(e)
        k_val = kemeny_after_edge_removals(G, removed_so_far, recompute_on_largest)
        history.append(k_val)

    current_k = kemeny_after_edge_removals(G, removed_so_far, recompute_on_largest)
    return EdgeKemenyResult(kemeny=current_k, removed_edges=removed_so_far, history=history)


if __name__ == "__main__":
    G = nx.cycle_graph(5)
    print("Kemeny constant:", kemeny_constant(G))

    node_result = interactive_kemeny(G, [0, 1])
    print(node_result)

    edge_result = interactive_kemeny_edges(G, [(0, 1), (2, 3)])
    print(edge_result)
