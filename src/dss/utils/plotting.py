"""Graph plotting utilities.

This module centralises the creation of network plots.  It uses
Matplotlib and NetworkX to draw graphs with configurable node sizes
and colours.  Because Streamlit caches Matplotlib figures, the layout
coordinates can be reused across multiple plots for consistency.
"""

from typing import Dict, Iterable, Optional, Any
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_network(
    G: nx.Graph,
    pos: Dict[Any, np.ndarray],
    *,
    node_size: Optional[Dict[Any, float]] = None,
    node_color: Optional[Dict[Any, float]] = None,
    cmap: str = "viridis",
    highlight_nodes: Optional[Iterable[Any]] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Render a network plot using a fixed layout.

    Parameters
    ----------
    G: networkx.Graph
        The graph to draw.
    pos: dict
        Precomputed layout positions for each node.
    node_size: dict, optional
        A mapping from node to size.  If provided, node sizes are
        scaled relative to the maximum value in the mapping.
    node_color: dict, optional
        A mapping from node to a numeric colour value.  If provided,
        colours are mapped through the given colormap.
    cmap: str, optional
        Name of a Matplotlib colormap to use when mapping numeric values
        to colours; defaults to "viridis".
    highlight_nodes: iterable, optional
        A collection of nodes to draw with a distinct border.
    title: str, optional
        Title for the plot.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plot.
    """

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_axis_off()
    # Prepare node sizes
    if node_size is not None:
        sizes = np.array([node_size.get(n, 1.0) for n in G.nodes()], dtype=float)
        if sizes.max() > 0:
            sizes = 300.0 * (sizes / sizes.max())  # scale to reasonable range
        else:
            sizes = np.full_like(sizes, 100.0)
    else:
        sizes = np.full(G.number_of_nodes(), 100.0)
    # Prepare node colours
    if node_color is not None:
        values = np.array([node_color.get(n, 0.0) for n in G.nodes()], dtype=float)
        vmin, vmax = values.min(), values.max()
        if vmin == vmax:
            vmin, vmax = 0.0, 1.0
        colours = values
    else:
        colours = np.zeros(G.number_of_nodes())
        vmin, vmax = 0.0, 1.0
    # Draw nodes
    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        node_size=sizes,
        node_color=colours,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
    )
    if highlight_nodes:
        # Draw highlighted nodes with red edgecolour
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=list(highlight_nodes),
            node_size=[sizes[list(G.nodes()).index(n)] for n in highlight_nodes],
            node_color=[colours[list(G.nodes()).index(n)] for n in highlight_nodes],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            edgecolors="red",
            linewidths=2.0,
            ax=ax,
        )
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)
    # Optionally draw labels on hover; labels not drawn by default
    if title:
        ax.set_title(title)
    return fig


if __name__ == "__main__":
    # Simple demonstration
    import networkx as nx
    G = nx.cycle_graph(5)
    pos = nx.spring_layout(G, seed=42)
    fig = plot_network(G, pos, title="Cycle graph")
    fig.savefig("_demo_plot.png")
    print("Demo plot saved to _demo_plot.png")