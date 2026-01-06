"""Graph validation utilities.

These functions perform sanity checks on adjacency matrices and graphs
constructed from them.  They ensure that the uploaded `.mtx` file
represents a valid undirected network and compute summary statistics.
"""

from typing import Tuple, List, Dict
import numpy as np
import networkx as nx


def is_symmetric(adjacency: np.ndarray, tol: float = 1e-8) -> bool:
    """Check whether a dense adjacency matrix is symmetric.

    Parameters
    ----------
    adjacency: ndarray
        A square dense matrix representing adjacency weights.
    tol: float, optional
        Tolerance for equality; defaults to 1e-8.

    Returns
    -------
    bool
        True if `adjacency` is symmetric within tolerance; False otherwise.
    """
    return np.allclose(adjacency, adjacency.T, atol=tol)


def has_self_loops(G: nx.Graph) -> bool:
    """Determine whether the graph contains any self loops."""
    return any(node in nbrs for node, nbrs in G.adj.items())


def component_sizes(G: nx.Graph) -> List[int]:
    """Return a list with the sizes of connected components."""
    return [len(c) for c in nx.connected_components(G)]


def basic_stats(G: nx.Graph) -> Dict[str, float]:
    """Compute basic network statistics.

    Parameters
    ----------
    G: networkx.Graph
        The graph to analyse.

    Returns
    -------
    dict
        A dictionary with keys `N` (number of nodes), `E` (number of edges),
        `density` (edge density) and `components` (number of connected
        components).
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = 0.0
    if n > 1:
        density = (2.0 * m) / (n * (n - 1))
    return {
        "N": n,
        "E": m,
        "density": density,
        "components": nx.number_connected_components(G),
    }


def validate_graph(G: nx.Graph) -> Dict[str, bool]:
    """Run validation checks on the graph.

    Parameters
    ----------
    G: networkx.Graph
        The graph to validate.

    Returns
    -------
    dict
        A dictionary with keys `symmetric`, `self_loops` and `connected`.
    """
    stats = {
        "symmetric": True,
        "self_loops": False,
        "connected": True,
    }
    # For symmetry, convert adjacency to dense
    A = nx.to_numpy_array(G)
    stats["symmetric"] = is_symmetric(A)
    stats["self_loops"] = has_self_loops(G)
    stats["connected"] = nx.is_connected(G)
    return stats


if __name__ == "__main__":
    # Quick self‑test on a simple graph
    import networkx as nx

    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    print("Validation:", validate_graph(G))
    print("Basic stats:", basic_stats(G))