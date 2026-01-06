"""Compute the Kemeny constant of a graph and evaluate its sensitivity."""

from typing import List, Dict, Any, Optional
import numpy as np
import networkx as nx

from ..types import KemenyResult
from ..logging_config import get_logger

logger = get_logger(__name__)


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
    if not G.is_directed() and nx.is_connected(G):
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
        pi_next = pi_next / pi_next.sum()
        if np.allclose(pi, pi_next, atol=1e-10):
            break
        pi = pi_next
    return pi


def kemeny_constant(G: nx.Graph) -> float:
    """Compute the Kemeny constant of the graph.

    The Kemeny constant is defined as `K = trace(Z)`, where
    `Z = (I - P + 1π)^{-1}` is the fundamental matrix of the Markov chain
    defined by the transition matrix `P` and stationary distribution `π`【429076285098930†L81-L94】.  For disconnected graphs
    the constant is computed on the largest connected component to ensure
    the chain is ergodic.

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
    # Use largest connected component if disconnected
    if not nx.is_connected(G):
        components = sorted(nx.connected_components(G), key=len, reverse=True)
        largest = G.subgraph(components[0]).copy()
        return kemeny_constant(largest)
    P = _transition_matrix(G)
    pi = _stationary_distribution(G, P)
    n = G.number_of_nodes()
    I = np.eye(n)
    # Fundamental matrix: (I - P + 1π)^(-1)
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
    """Compute Kemeny constant after removing specified nodes.

    Parameters
    ----------
    G: networkx.Graph
        Original graph.
    removed_nodes: list
        Nodes to remove before computing the Kemeny constant.
    recompute_on_largest: bool, optional
        If True, compute the constant only on the largest connected
        component when the resulting graph is disconnected.  Otherwise,
        return NaN when the graph is disconnected.

    Returns
    -------
    float
        The Kemeny constant of the pruned graph or NaN if undefined.
    """
    H = G.copy()
    H.remove_nodes_from(removed_nodes)
    if H.number_of_nodes() == 0:
        return float("nan")
    if nx.is_connected(H):
        return kemeny_constant(H)
    if recompute_on_largest:
        components = sorted(nx.connected_components(H), key=len, reverse=True)
        largest = H.subgraph(components[0]).copy()
        return kemeny_constant(largest)
    else:
        return float("nan")


def interactive_kemeny(
    G: nx.Graph,
    selected_nodes: List[Any],
    recompute_on_largest: bool = True,
) -> KemenyResult:
    """Return a KemenyResult tracking removals and Kemeny values.

    Parameters
    ----------
    G: networkx.Graph
        The original graph.
    selected_nodes: list
        Nodes selected for removal, in the order they were selected.
    recompute_on_largest: bool, optional
        Whether to compute on largest component if disconnected.

    Returns
    -------
    KemenyResult
        Object containing the current Kemeny constant, list of removed
        nodes and history of Kemeny constants after each removal.
    """
    history = []
    removed_so_far: List[Any] = []
    for node in selected_nodes:
        removed_so_far.append(node)
        k_val = kemeny_after_removals(G, removed_so_far, recompute_on_largest)
        history.append(k_val)
    current_k = kemeny_after_removals(G, removed_so_far, recompute_on_largest)
    return KemenyResult(kemeny=current_k, removed_nodes=removed_so_far, history=history)


if __name__ == "__main__":
    import networkx as nx
    G = nx.cycle_graph(5)
    print("Kemeny constant:", kemeny_constant(G))
    result = interactive_kemeny(G, [0, 1])
    print(result)