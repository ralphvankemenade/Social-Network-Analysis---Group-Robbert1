"""Community detection algorithms and summaries."""

from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.cluster import SpectralClustering
from networkx.algorithms.community import girvan_newman
from networkx.algorithms.community.quality import modularity as nx_modularity
import community as community_louvain  # python-louvain package

from ..types import CommunityResult
from ..logging_config import get_logger

logger = get_logger(__name__)


def _louvain_communities(G: nx.Graph) -> Dict[Any, int]:
    """Detect communities using the Louvain method."""
    partition = community_louvain.best_partition(G)
    return partition


def _girvan_newman_communities(G: nx.Graph, k: Optional[int] = None) -> Dict[Any, int]:
    """Detect communities using Girvan–Newman divisive algorithm.

    Parameters
    ----------
    G: networkx.Graph
        The graph to cluster.
    k: int, optional
        Desired number of clusters.  If None, uses the first split that
        yields at least two communities.

    Returns
    -------
    dict
        Mapping from node to community index.
    """
    comp_gen = girvan_newman(G)
    communities = None
    if k is None or k < 2:
        # Take the first split
        communities = next(comp_gen)
    else:
        for _ in range(k - 1):
            communities = next(comp_gen)
    # communities is a tuple of sets
    comm_list = list(communities)
    labels = {}
    for idx, comm in enumerate(comm_list):
        for node in comm:
            labels[node] = idx
    return labels


def _spectral_communities(G: nx.Graph, k: int = 2) -> Dict[Any, int]:
    """Detect communities using spectral clustering on the Laplacian."""
    n = G.number_of_nodes()
    if n <= k:
        # Assign each node to its own community
        return {node: i for i, node in enumerate(G.nodes())}
    L = nx.normalized_laplacian_matrix(G).todense()
    # Use spectral clustering on the Laplacian (affinity is negative distance)
    sc = SpectralClustering(n_clusters=k, affinity="precomputed", assign_labels="kmeans", random_state=0)
    # Convert Laplacian to similarity by subtracting from max
    lap = np.asarray(L, dtype=float)
    max_val = lap.max()
    affinity = max_val - lap
    labels_array = sc.fit_predict(affinity)
    labels = {node: int(labels_array[i]) for i, node in enumerate(G.nodes())}
    return labels


def _compute_summary(G: nx.Graph, labels: Dict[Any, int]) -> pd.DataFrame:
    """Compute cluster size and within‑edge ratio for each community."""
    df = pd.DataFrame({"community": [labels[node] for node in G.nodes()]}, index=list(G.nodes()))
    # Size of each community
    sizes = df.groupby("community").size().rename("size")
    # Compute within‑edge ratio: number of internal edges divided by number of
    # possible internal edges for each community
    ratios = []
    for comm in sizes.index:
        nodes = df.index[df["community"] == comm].tolist()
        subgraph = G.subgraph(nodes)
        n = subgraph.number_of_nodes()
        m = subgraph.number_of_edges()
        possible = n * (n - 1) / 2 if n > 1 else 1
        ratios.append(m / possible if possible > 0 else 0.0)
    summary = pd.DataFrame({"size": sizes, "within_ratio": ratios})
    return summary


def compute_communities(
    G: nx.Graph,
    method: str = "louvain",
    k: Optional[int] = None,
) -> CommunityResult:
    """Detect communities in the graph using the specified method.

    Parameters
    ----------
    G: networkx.Graph
        The graph to cluster.
    method: str, optional
        Community detection algorithm: "louvain", "girvan_newman", or "spectral".
    k: int, optional
        Desired number of clusters (only used for Girvan–Newman and spectral).

    Returns
    -------
    CommunityResult
        A container with labels, modularity and summary statistics.
    """
    if method == "louvain":
        labels = _louvain_communities(G)
    elif method == "girvan_newman":
        labels = _girvan_newman_communities(G, k)
    elif method == "spectral":
        k_eff = k or 2
        labels = _spectral_communities(G, k_eff)
    else:
        raise ValueError(f"Unknown community detection method: {method}")
    # Convert labels to a list of sets for modularity calculation
    communities = []
    label_to_nodes: Dict[int, List[Any]] = {}
    for node, lab in labels.items():
        label_to_nodes.setdefault(lab, []).append(node)
    communities = [set(nodes) for nodes in label_to_nodes.values()]
    try:
        Q = nx_modularity(G, communities)
    except Exception as e:
        logger.warning(f"Failed to compute modularity: {e}")
        Q = 0.0
    summary = _compute_summary(G, labels)
    return CommunityResult(labels=labels, modularity=Q, summary=summary)


if __name__ == "__main__":
    import networkx as nx
    G = nx.karate_club_graph()
    result = compute_communities(G, method="louvain")
    print(result.modularity)
    print(result.summary)