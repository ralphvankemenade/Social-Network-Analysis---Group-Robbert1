"""Centrality measures and aggregation.

This module computes a variety of centrality measures using NetworkX
functions and provides helper functions to combine them.  Degree,
Katz, eigenvector, betweenness, closeness and PageRank centralities are
supported.  Additional measures can be added easily by extending
`compute_centralities`.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import networkx as nx

from ..types import CentralityResult
from ..logging_config import get_logger

logger = get_logger(__name__)


def _safe_katz_centrality(G: nx.Graph, alpha: Optional[float] = None, beta: float = 1.0) -> Dict[Any, float]:
    """Compute Katz centrality with a safe alpha.

    If `alpha` is None, estimate a stable value based on the largest
    eigenvalue of the adjacency matrix.  NetworkX’s `katz_centrality`
    requires that `alpha < 1/λ_max`, where λ_max is the largest eigenvalue
    in magnitude.  When the graph is disconnected, `lambda_max` is
    approximated via the spectral norm of the adjacency matrix.
    """
    try:
        # Compute largest eigenvalue of adjacency matrix
        A = nx.to_scipy_sparse_matrix(G, dtype=float)
        # Use power iteration to estimate spectral radius
        # Compute singular values; spectral radius <= largest singular value
        u, s, vt = np.linalg.svd(A.toarray(), full_matrices=False)
        lambda_max = s[0]
        if lambda_max == 0:
            lambda_max = 1.0
    except Exception as e:
        logger.warning(f"Failed to estimate spectral radius: {e}")
        lambda_max = 1.0
    if alpha is None:
        alpha = 0.9 / lambda_max
    try:
        return nx.katz_centrality_numpy(G, alpha=alpha, beta=beta, normalized=True)
    except Exception as e:
        logger.warning(f"Katz centrality failed: {e}")
        # Fall back to zeros
        return {n: 0.0 for n in G.nodes()}


def compute_centralities(G: nx.Graph) -> pd.DataFrame:
    """Compute multiple centrality measures for the graph.

    Parameters
    ----------
    G: networkx.Graph
        The graph for which to compute centralities.

    Returns
    -------
    pandas.DataFrame
        A table indexed by node with one column per centrality measure.
    """
    # Degree centrality
    deg = nx.degree_centrality(G)
    # Katz centrality with auto‑chosen alpha
    katz = _safe_katz_centrality(G)
    # Eigenvector centrality
    try:
        eigen = nx.eigenvector_centrality_numpy(G)
    except Exception as e:
        logger.warning(f"Eigenvector centrality failed: {e}")
        eigen = {n: 0.0 for n in G.nodes()}
    # Betweenness centrality
    between = nx.betweenness_centrality(G, normalized=True)
    # Closeness centrality
    closeness = nx.closeness_centrality(G)
    # PageRank (as an additional robustness check)
    pagerank = nx.pagerank(G)
    # Construct DataFrame
    df = pd.DataFrame(
        {
            "degree": pd.Series(deg),
            "katz": pd.Series(katz),
            "eigenvector": pd.Series(eigen),
            "betweenness": pd.Series(between),
            "closeness": pd.Series(closeness),
            "pagerank": pd.Series(pagerank),
        }
    )
    return df


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise each column of the DataFrame to the range [0, 1]."""
    normed = (df - df.min()) / (df.max() - df.min()).replace([np.inf, -np.inf], 0.0)
    normed = normed.fillna(0.0)
    return normed


def combine_centralities(df: pd.DataFrame, weights: Optional[Dict[str, float]] = None) -> pd.Series:
    """Combine centrality measures using a weighted sum of normalised scores.

    Parameters
    ----------
    df: pandas.DataFrame
        DataFrame of centrality measures with nodes as index.
    weights: dict, optional
        Mapping from column name to weight.  If `None`, all columns are
        weighted equally.

    Returns
    -------
    pandas.Series
        Combined scores for each node.
    """
    normed = _normalise_columns(df)
    if weights is None:
        weights = {col: 1.0 for col in df.columns}
    # Ensure weights sum to 1 for interpretability
    total_weight = sum(weights.values())
    if total_weight == 0:
        weights = {col: 1.0 for col in df.columns}
        total_weight = float(len(df.columns))
    weights = {col: w / total_weight for col, w in weights.items()}
    # Weighted sum
    combined = sum(normed[col] * weights.get(col, 0.0) for col in df.columns)
    return combined


def borda_count(df: pd.DataFrame) -> pd.Series:
    """Aggregate rankings via Borda count.

    Each centrality measure induces a ranking.  Nodes receive points
    according to their rank (higher points for higher rank).  The sum of
    points across measures yields an aggregated ranking.  Ties are
    handled by assigning the average rank.  The returned Series has
    smaller values for better ranks.
    """
    ranks = df.rank(ascending=False, method="average")
    borda_scores = ranks.mean(axis=1)
    return borda_scores


def compute_centrality_result(G: nx.Graph, weights: Optional[Dict[str, float]] = None) -> CentralityResult:
    """Compute centralities and return a CentralityResult.

    Parameters
    ----------
    G: networkx.Graph
        The graph to analyse.
    weights: dict, optional
        Weights for combining centralities.

    Returns
    -------
    CentralityResult
        A structured container of centrality data.
    """
    table = compute_centralities(G)
    combined = combine_centralities(table, weights)
    ranks = table.rank(ascending=False, method="min")
    return CentralityResult(table=table, combined_scores=combined, ranks=ranks)


if __name__ == "__main__":
    # Example usage on a small graph
    G = nx.path_graph(4)
    result = compute_centrality_result(G)
    print(result.table)
    print("Combined scores:")
    print(result.combined_scores)