"""Graph plotting utilities.

This module centralises the creation of network plots.  It uses
Matplotlib and NetworkX to draw graphs with configurable node sizes
and colours.  Because Streamlit caches Matplotlib figures, the layout
coordinates can be reused across multiple plots for consistency.
"""

from typing import Any, Dict, Iterable, Optional, Tuple, List
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# def plot_network(
#     G: nx.Graph,
#     pos: Dict[Any, np.ndarray],
#     *,
#     node_size: Optional[Dict[Any, float]] = None,
#     node_color: Optional[Dict[Any, float]] = None,
#     cmap: str = "viridis",
#     highlight_nodes: Optional[Iterable[Any]] = None,
#     title: Optional[str] = None,
#     show_labels: bool = True,
#     label_dict: Optional[Dict[Any, str]] = None,
# ) -> plt.Figure:
#     """Render a network plot using a fixed layout with optional labels.

#     Parameters
#     ----------
#     G: networkx.Graph
#         The graph to draw.
#     pos: dict
#         Precomputed layout positions for each node.
#     node_size: dict, optional
#         A mapping from node to size.  If provided, node sizes are
#         scaled relative to the maximum value in the mapping.
#     node_color: dict, optional
#         A mapping from node to a numeric colour value.  If provided,
#         colours are mapped through the given colormap.
#     cmap: str, optional
#         Name of a Matplotlib colormap to use when mapping numeric values
#         to colours; defaults to ``"viridis"``.
#     highlight_nodes: iterable, optional
#         A collection of nodes to draw with a distinct border.
#     title: str, optional
#         Title for the plot.
#     show_labels: bool, optional
#         If True, draw node labels (usually the node identifiers) on the
#         graph.  Font sizes will be scaled with node size to keep
#         proportions.
#     label_dict: dict, optional
#         An optional mapping of nodes to label strings.  If None,
#         ``str(node)`` is used for each node.

#     Returns
#     -------
#     matplotlib.figure.Figure
#         The figure containing the plot.
#     """

#     # Create a square figure for consistent aspect ratio
#     fig, ax = plt.subplots(figsize=(6, 6))
#     ax.set_axis_off()
#     # Prepare node sizes.  Use the raw values to compute scaling and
#     # later derive font sizes.  If not provided, default to uniform size.
#     if node_size is not None:
#         sizes_raw = np.array([node_size.get(n, 1.0) for n in G.nodes()], dtype=float)
#         if sizes_raw.max() > 0:
#             sizes = 300.0 * (sizes_raw / sizes_raw.max())
#         else:
#             sizes = np.full_like(sizes_raw, 100.0)
#     else:
#         sizes_raw = np.ones(G.number_of_nodes(), dtype=float)
#         sizes = np.full(G.number_of_nodes(), 100.0)
#     # Prepare node colours.  Normalize values to a colour scale.  If
#     # uniform or unspecified, assign a constant zero value to each node.
#     if node_color is not None:
#         values = np.array([node_color.get(n, 0.0) for n in G.nodes()], dtype=float)
#         vmin, vmax = values.min(), values.max()
#         if vmin == vmax:
#             vmin, vmax = 0.0, 1.0
#         colours = values
#     else:
#         colours = np.zeros(G.number_of_nodes())
#         vmin, vmax = 0.0, 1.0
#     # Draw nodes with sizes and colours
#     nx.draw_networkx_nodes(
#         G,
#         pos,
#         node_size=sizes,
#         node_color=colours,
#         cmap=cmap,
#         vmin=vmin,
#         vmax=vmax,
#         ax=ax,
#     )
#     # Highlight specified nodes by re‑drawing them with a red border
#     if highlight_nodes:
#         nodelist = list(highlight_nodes)
#         highlight_sizes = [sizes[list(G.nodes()).index(n)] for n in nodelist]
#         highlight_colours = [colours[list(G.nodes()).index(n)] for n in nodelist]
#         nx.draw_networkx_nodes(
#             G,
#             pos,
#             nodelist=nodelist,
#             node_size=highlight_sizes,
#             node_color=highlight_colours,
#             cmap=cmap,
#             vmin=vmin,
#             vmax=vmax,
#             edgecolors="red",
#             linewidths=2,
#             ax=ax,
#         )
#     # Draw edges with light transparency
#     nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)
#     # Draw labels if requested.  Font sizes are scaled between 6 and 12
#     # based on the relative node size.  If a custom ``label_dict`` is
#     # provided, its values are used; otherwise use the node identifier.
#     if show_labels:
#         max_raw = sizes_raw.max() if sizes_raw.max() > 0 else 1.0
#         for idx, n in enumerate(G.nodes()):
#             x, y = pos[n]
#             # Scale font size: base of 6 plus up to 6 additional points
#             # fs = 6 + 6 * (sizes_raw[idx] / max_raw)
#             fs = 4 + 3 * (sizes_raw[idx] / max_raw)
#             label = label_dict[n] if (label_dict is not None and n in label_dict) else str(n)
#             ax.text(x, y, label, fontsize=fs, ha='center', va='center', color='white') #,bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=0.5)) #color='black')
#     # Set title if provided
#     if title:
#         ax.set_title(title)
#     return fig
# --- plot_network (UPDATED) ---

def plot_network(
    G: nx.Graph,
    pos: Dict[Any, np.ndarray],
    *,
    node_size: Optional[Dict[Any, float]] = None,
    node_color: Optional[Dict[Any, float]] = None,
    cmap: str = "viridis",
    highlight_nodes: Optional[Iterable[Any]] = None,
    title: Optional[str] = None,
    show_labels: bool = True,
    label_dict: Optional[Dict[Any, str]] = None,
    removed_edges: Optional[Iterable[Tuple[Any, Any]]] = None,
) -> plt.Figure:
    """Render a network plot using a fixed layout with optional labels.

    Parameters
    ----------
    G: networkx.Graph
        The graph to draw.
    pos: dict
        Precomputed layout positions for each node.
    node_size: dict, optional
        A mapping from node to size. If provided, node sizes are
        scaled relative to the maximum value in the mapping.
    node_color: dict, optional
        A mapping from node to a numeric color value. If provided,
        colors are mapped through the given colormap.
    cmap: str, optional
        Name of a Matplotlib colormap to use when mapping numeric values
        to colors; defaults to ``"viridis"``.
    highlight_nodes: iterable, optional
        A collection of nodes to draw with a distinct border.
    title: str, optional
        Title for the plot.
    show_labels: bool, optional
        If True, draw node labels (usually the node identifiers) on the
        graph. Font sizes will be scaled with node size to keep
        proportions.
    label_dict: dict, optional
        An optional mapping of nodes to label strings. If None,
        ``str(node)`` is used for each node.
    removed_edges: iterable of (u, v), optional
        Edges to overlay as visually "removed" (drawn as dashed red lines).
        Useful when you want to keep the overall structure visible while
        clearly indicating which connections were removed.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plot.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_axis_off()

    # Prepare node sizes
    if node_size is not None:
        sizes_raw = np.array([node_size.get(n, 1.0) for n in G.nodes()], dtype=float)
        if sizes_raw.max() > 0:
            sizes = 300.0 * (sizes_raw / sizes_raw.max())
        else:
            sizes = np.full_like(sizes_raw, 100.0)
    else:
        sizes_raw = np.ones(G.number_of_nodes(), dtype=float)
        sizes = np.full(G.number_of_nodes(), 100.0)
    # Prepare node colors
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
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=sizes,
        node_color=colours,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
    )
    # Highlight nodes
    if highlight_nodes:
        nodelist = list(highlight_nodes)
        highlight_sizes = [sizes[list(G.nodes()).index(n)] for n in nodelist]
        highlight_colours = [colours[list(G.nodes()).index(n)] for n in nodelist]
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodelist,
            node_size=highlight_sizes,
            node_color=highlight_colours,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            edgecolors="red",
            linewidths=2,
            ax=ax,
        )
    # Draw edges (base)
    nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)
    # Overlay removed edges as dashed red (drawn on top)
    if removed_edges:
        dashed: List[Tuple[Any, Any]] = list(removed_edges)
        if not G.is_directed():
            s = set(dashed)
            s |= {(v, u) for (u, v) in s}
            dashed = list(s)
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=dashed,
            ax=ax,
            edge_color="red",
            style="dashed",
            width=1,
            alpha=0.6,
        )
    # Labels
    if show_labels:
        max_raw = sizes_raw.max() if sizes_raw.max() > 0 else 1.0
        for idx, n in enumerate(G.nodes()):
            x, y = pos[n]
            fs = 4 + 3 * (sizes_raw[idx] / max_raw)
            label = label_dict[n] if (label_dict is not None and n in label_dict) else str(n)
            ax.text(x, y, label, fontsize=fs, ha="center", va="center", color="white")
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
