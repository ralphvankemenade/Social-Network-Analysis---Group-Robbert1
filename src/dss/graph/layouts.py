"""Compute and cache graph layouts."""

from typing import Dict, Any
import networkx as nx
import numpy as np
from ..utils.caching import cache_data
from ..config import DEFAULTS


@cache_data
def compute_layout(G: nx.Graph, layout_name: str = DEFAULTS.layout, seed: int = DEFAULTS.seed) -> Dict[Any, np.ndarray]:
    """Compute a layout for visualising the graph.

    Parameters
    ----------
    G: networkx.Graph
        The graph to layout.
    layout_name: str, optional
        The name of the layout algorithm.  Currently only "spring" is
        supported.  Additional layouts could be added in the future.
    seed: int, optional
        Seed for random layouts, ensuring reproducibility.

    Returns
    -------
    dict
        A mapping of node to 2‑D coordinates.
    """
    if layout_name == "spring":
        return nx.spring_layout(G, seed=seed)
    elif layout_name == "kamada_kawai":
        return nx.kamada_kawai_layout(G)
    else:
        # Fallback to spring layout
        return nx.spring_layout(G, seed=seed)


if __name__ == "__main__":
    import networkx as nx
    G = nx.cycle_graph(5)
    pos = compute_layout(G)
    print(pos)