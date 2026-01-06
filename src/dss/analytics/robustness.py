"""Robustness analysis for community detection."""

from typing import List
import random
import networkx as nx
from sklearn.metrics import adjusted_rand_score

from ..types import RobustnessResult
from .communities import compute_communities
from ..logging_config import get_logger

logger = get_logger(__name__)


def perturbation_test(
    G: nx.Graph,
    method: str = "louvain",
    p: float = 0.05,
    runs: int = 50,
    k: int = 2,
) -> RobustnessResult:
    """Evaluate robustness of community detection by removing random edges.

    Parameters
    ----------
    G: networkx.Graph
        The original graph.
    method: str, optional
        Community detection method (see `compute_communities`).
    p: float, optional
        Fraction of edges to remove in each perturbation.  Must be in (0, 1).
    runs: int, optional
        Number of perturbation trials.
    k: int, optional
        Number of clusters for methods that require it (Girvan–Newman,
        spectral).

    Returns
    -------
    RobustnessResult
        A container of ARI scores and modularity drops across runs.
    """
    # Original community detection
    orig_comm = compute_communities(G, method=method, k=k)
    orig_labels = [orig_comm.labels[node] for node in G.nodes()]
    orig_mod = orig_comm.modularity
    ari_scores: List[float] = []
    mod_drops: List[float] = []
    edges = list(G.edges())
    m = len(edges)
    remove_count = max(1, int(p * m))
    for _ in range(runs):
        # Create a copy and remove random edges
        H = G.copy()
        removed = random.sample(edges, remove_count)
        H.remove_edges_from(removed)
        # Compute communities on perturbed graph
        try:
            pert_comm = compute_communities(H, method=method, k=k)
            pert_labels = [pert_comm.labels.get(node, -1) for node in G.nodes()]
            ari = adjusted_rand_score(orig_labels, pert_labels)
            mod_drop = orig_mod - pert_comm.modularity
        except Exception as e:
            logger.warning(f"Perturbation run failed: {e}")
            ari = 0.0
            mod_drop = 0.0
        ari_scores.append(ari)
        mod_drops.append(mod_drop)
    return RobustnessResult(ari_scores=ari_scores, modularity_drops=mod_drops)


if __name__ == "__main__":
    import networkx as nx
    G = nx.karate_club_graph()
    result = perturbation_test(G, method="louvain", runs=5)
    print(result)