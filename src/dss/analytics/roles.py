"""Role similarity and clustering based on Cooper & Barahona (2010).

This module constructs structural signatures for each node in a network
and computes pairwise similarities.  It then clusters nodes into roles
based on these similarities using either spectral or hierarchical
clustering.  A summary of each role cluster is provided in terms of
basic centrality statistics.
"""

from typing import Dict, Any, Optional, Tuple, Callable
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import pairwise_distances
from sklearn.cluster import SpectralClustering, AgglomerativeClustering

from ..types import RoleResult
from .centrality import compute_centralities
from ..logging_config import get_logger

logger = get_logger(__name__)


def _k_hop_signature(G: nx.Graph, k: int = 3) -> np.ndarray:
    """Compute k‑hop degree distribution signatures for each node.

    For each node, a vector of length `k` is created where the `i`‑th
    element counts the number of neighbours at distance exactly `i+1`.

    Parameters
    ----------
    G: networkx.Graph
        The graph to analyse.
    k: int, optional
        The maximum hop distance to include; defaults to 3.

    Returns
    -------
    numpy.ndarray
        An array of shape (n_nodes, k) containing the signatures.
    """
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    index_map = {node: idx for idx, node in enumerate(nodes)}
    signatures = np.zeros((n, k), dtype=float)
    # Precompute shortest path lengths
    lengths = dict(nx.all_pairs_shortest_path_length(G, cutoff=k))
    for node, dist_dict in lengths.items():
        row = index_map[node]
        for target, dist in dist_dict.items():
            if dist == 0 or dist > k:
                continue
            signatures[row, dist - 1] += 1
    return signatures


def _random_walk_profiles(G: nx.Graph, t: int = 3) -> np.ndarray:
    """Compute random‑walk probability profiles after `t` steps for each node.

    The transition matrix `P` is built by normalising rows of the adjacency
    matrix.  The `t`‑step transition probabilities are obtained from `P^t`.

    Parameters
    ----------
    G: networkx.Graph
        The graph to analyse.
    t: int, optional
        Number of steps for the random walk; defaults to 3.

    Returns
    -------
    numpy.ndarray
        An array of shape (n_nodes, n_nodes) where each row `i` contains the
        probability distribution of being at each node after `t` steps starting
        from node `i`.
    """
    # Build transition matrix
    nodes = list(G.nodes())
    idx_map = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    P = np.zeros((n, n), dtype=float)
    for i, u in enumerate(nodes):
        deg = G.degree(u)
        if deg > 0:
            for v in G.neighbors(u):
                j = idx_map[v]
                P[i, j] = 1.0 / deg
    # Compute P^t
    Pt = np.linalg.matrix_power(P, t)
    return Pt


def _compute_similarity_matrix(
    features: np.ndarray,
    metric: str = "cosine",
) -> np.ndarray:
    """Compute a similarity matrix from feature vectors.

    Parameters
    ----------
    features: ndarray
        Array of shape (n_samples, n_features) containing feature vectors.
    metric: str, optional
        Similarity measure: "cosine" or "correlation".

    Returns
    -------
    ndarray
        A square matrix of pairwise similarities in the range [0, 1].
    """
    # Compute pairwise distances; convert to similarity
    if metric == "cosine":
        distances = pairwise_distances(features, metric="cosine")
    elif metric == "correlation":
        distances = pairwise_distances(features, metric="correlation")
    else:
        raise ValueError(f"Unknown similarity metric: {metric}")
    # Similarity = 1 - distance, but clip to [0, 1]
    sim = 1.0 - distances
    sim = np.clip(sim, 0.0, 1.0)
    return sim


def _cluster_similarity_matrix(
    similarity: np.ndarray,
    n_clusters: Optional[int] = None,
    method: str = "spectral",
) -> List[int]:
    """Cluster a similarity matrix and return labels.

    Parameters
    ----------
    similarity: ndarray
        Square similarity matrix.
    n_clusters: int, optional
        Desired number of clusters.  If None, defaults to the square root
        of the number of nodes rounded up.
    method: str, optional
        "spectral" for SpectralClustering or "hierarchical" for
        AgglomerativeClustering.

    Returns
    -------
    list of int
        Cluster labels for each sample.
    """
    n = similarity.shape[0]
    if n_clusters is None:
        n_clusters = max(2, int(np.ceil(np.sqrt(n))))
    if method == "spectral":
        # Spectral clustering expects an affinity (similarity) matrix
        sc = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=0,
        )
        labels = sc.fit_predict(similarity)
    elif method == "hierarchical":
        hc = AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            linkage="average",
        )
        labels = hc.fit_predict(1.0 - similarity)
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    return labels.tolist()


def compute_roles(
    G: nx.Graph,
    signature: str = "k-hop",
    k: int = 3,
    t: int = 3,
    similarity_metric: str = "cosine",
    clustering_method: str = "spectral",
    n_clusters: Optional[int] = None,
    centrality_table: Optional[pd.DataFrame] = None,
) -> RoleResult:
    """Compute role similarity, cluster roles and summarise clusters.

    Parameters
    ----------
    G: networkx.Graph
        Graph on which to compute roles.
    signature: str, optional
        Type of structural signature: "k-hop" or "random-walk".
    k: int, optional
        Maximum hop for k-hop signatures; ignored for random-walk.
    t: int, optional
        Number of steps for random-walk profiles; ignored for k-hop.
    similarity_metric: str, optional
        Similarity metric: "cosine" or "correlation".
    clustering_method: str, optional
        Clustering algorithm: "spectral" or "hierarchical".
    n_clusters: int, optional
        Desired number of clusters; if None, uses sqrt heuristic.
    centrality_table: pandas.DataFrame, optional
        Precomputed centralities; if provided, cluster summaries will
        include mean centralities per role.

    Returns
    -------
    RoleResult
        A container with the similarity matrix, cluster labels and summary.
    """
    nodes = list(G.nodes())
    if signature == "k-hop":
        features = _k_hop_signature(G, k)
    elif signature == "random-walk":
        features = _random_walk_profiles(G, t)
    else:
        raise ValueError(f"Unsupported signature type: {signature}")
    similarity = _compute_similarity_matrix(features, metric=similarity_metric)
    labels_list = _cluster_similarity_matrix(similarity, n_clusters, method=clustering_method)
    # Build labels dictionary keyed by node
    labels = {node: labels_list[i] for i, node in enumerate(nodes)}
    # Compute summary statistics per role
    if centrality_table is None:
        centrality_table = compute_centralities(G)
    df = centrality_table.copy()
    df["role"] = [labels[node] for node in nodes]
    summary = df.groupby("role").mean()
    summary["size"] = df.groupby("role").size()
    # Create RoleResult
    return RoleResult(similarity_matrix=similarity, labels=labels, summary=summary)


if __name__ == "__main__":
    import networkx as nx
    G = nx.karate_club_graph()
    result = compute_roles(G, signature="k-hop", k=2, similarity_metric="cosine", clustering_method="spectral", n_clusters=4)
    print("Role labels:", result.labels)
    print("Summary:")
    print(result.summary)