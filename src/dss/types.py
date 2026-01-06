"""Data types and simple containers used in the DSS.

The DSS works with several complex return values (for example, centrality
tables, role clustering results and community detection outputs).  Defining
small data classes clarifies the shape of these objects and improves
type‑checking.  These classes should not contain heavy logic; they merely
organise data for consumption by Streamlit pages.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np


# Type alias for node identifiers.  NetworkX typically uses integers, but
# strings or other hashable types are also possible.
Node = Any


@dataclass
class CentralityResult:
    """Container for centrality metrics and combined scores."""

    table: pd.DataFrame  # DataFrame with one row per node and columns per metric
    combined_scores: pd.Series  # Weighted or aggregated scores
    ranks: pd.DataFrame  # Ranks per metric


@dataclass
class RoleResult:
    """Container for role similarity and clustering results."""

    similarity_matrix: np.ndarray  # Pairwise similarity matrix
    labels: Dict[Node, int]  # Role cluster label per node
    summary: pd.DataFrame  # Summary statistics per role cluster


@dataclass
class CommunityResult:
    """Container for community detection results."""

    labels: Dict[Node, int]  # Community label per node
    modularity: float  # Modularity score Q
    summary: pd.DataFrame  # Cluster size and intra‑edge ratio per community


@dataclass
class RobustnessResult:
    """Container for perturbation robustness results."""

    ari_scores: List[float]  # Adjusted Rand Index values across runs
    modularity_drops: List[float]  # Drop in modularity relative to original


@dataclass
class KemenyResult:
    """Container for Kemeny constant calculations."""

    kemeny: float  # Current value of the Kemeny constant
    removed_nodes: List[Node] = field(default_factory=list)  # Nodes removed so far
    history: List[float] = field(default_factory=list)  # Kemeny values after each removal


@dataclass
class ArrestAssignmentResult:
    """Container for arrest optimisation output."""

    assignment: Dict[Node, int]  # 0/1 department assignment per node
    objective: float  # Objective value of the optimisation
    cut_edges: int  # Number of edges crossing the departments
    effective_arrests: float  # Estimated number of effective arrests
    risk_edges: List[tuple]  # Edges whose endpoints are assigned to different departments


if __name__ == "__main__":
    # Example: create an empty centrality result for demonstration
    import networkx as nx
    import pandas as pd

    G = nx.path_graph(3)
    df = pd.DataFrame(
        {
            "degree": [1.0, 2.0, 1.0],
            "eigenvector": [0.5, 1.0, 0.5],
        },
        index=list(G.nodes()),
    )
    combined = df.mean(axis=1)
    ranks = df.rank(ascending=False, method="min")
    result = CentralityResult(table=df, combined_scores=combined, ranks=ranks)
    print(result)