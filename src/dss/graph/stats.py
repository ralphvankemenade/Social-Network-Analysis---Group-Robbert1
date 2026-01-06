"""Compute statistics about graphs."""

from typing import Dict, Any, List
import networkx as nx


def number_of_nodes(G: nx.Graph) -> int:
    """Return the number of nodes in the graph."""
    return G.number_of_nodes()


def number_of_edges(G: nx.Graph) -> int:
    """Return the number of edges in the graph."""
    return G.number_of_edges()


def density(G: nx.Graph) -> float:
    """Compute the edge density of the graph."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    return (2.0 * m) / (n * (n - 1)) if n > 1 else 0.0


def degree_vector(G: nx.Graph) -> List[int]:
    """Return a list of degrees for all nodes."""
    return [d for _, d in G.degree()]


def connected_component_sizes(G: nx.Graph) -> List[int]:
    """Return the sizes of connected components."""
    return [len(c) for c in nx.connected_components(G)]


def basic_statistics(G: nx.Graph) -> Dict[str, Any]:
    """Return a dictionary of basic network statistics."""
    return {
        "N": number_of_nodes(G),
        "E": number_of_edges(G),
        "density": density(G),
        "components": nx.number_connected_components(G),
        "component_sizes": connected_component_sizes(G),
    }


if __name__ == "__main__":
    import networkx as nx
    G = nx.path_graph(4)
    print(basic_statistics(G))