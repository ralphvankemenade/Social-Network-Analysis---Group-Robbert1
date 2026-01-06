"""Functions to build NetworkX graphs from adjacency matrices."""

from typing import Optional
import networkx as nx
from scipy.sparse import csr_matrix


def build_graph(adjacency: csr_matrix, directed: Optional[bool] = False) -> nx.Graph:
    """Construct a NetworkX graph from a sparse adjacency matrix.

    Parameters
    ----------
    adjacency: csr_matrix
        A square sparse matrix representing the adjacency weights between
        nodes.  Non‑zero entries indicate edges; diagonal entries are
        ignored.
    directed: bool, optional
        Whether to build a directed graph.  If False (default), an
        undirected graph is built.  Otherwise, a :class:`networkx.DiGraph` is returned.

    Returns
    -------
    networkx.Graph
        The resulting graph.  Self loops are removed automatically.

    Notes
    -----
    Recent versions of NetworkX (3.0 and above) have deprecated
    ``from_scipy_sparse_matrix`` in favour of ``from_scipy_sparse_array``【372834749795475†L130-L135】.
    To maintain compatibility across versions, this function manually
    constructs the graph by iterating over the non‑zero entries of the
    adjacency matrix.  This avoids reliance on deprecated API and
    eliminates the need for SciPy sparse array conversion.
    """
    n, m = adjacency.shape
    if n != m:
        raise ValueError("Adjacency matrix must be square.")
    # Remove self loops by setting the diagonal to zero
    adjacency = adjacency.tolil()
    adjacency.setdiag(0)
    adjacency = adjacency.tocsr()
    # Determine graph type
    if directed:
        G: nx.Graph = nx.DiGraph()
    else:
        G = nx.Graph()
    # Add edges for each non-zero entry
    rows, cols = adjacency.nonzero()
    for u, v in zip(rows, cols):
        # networkx will automatically collapse parallel edges for Graph
        # Self loops have been removed above
        G.add_edge(int(u), int(v))
    return G


if __name__ == "__main__":
    # Quick demonstration
    import numpy as np
    from scipy.sparse import csr_matrix

    adj = csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]))
    G = build_graph(adj)
    print("Nodes:", G.nodes())
    print("Edges:", list(G.edges()))