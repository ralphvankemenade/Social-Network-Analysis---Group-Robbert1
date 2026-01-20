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
#     edge_color: Optional[Dict[Tuple[Any, Any], float]] = None,
#     cmap: str = "viridis",
#     highlight_nodes: Optional[Iterable[Any]] = None,
#     title: Optional[str] = None,
#     show_labels: bool = True,
#     label_dict: Optional[Dict[Any, str]] = None,
#     removed_edges: Optional[Iterable[Tuple[Any, Any]]] = None,
# ) -> plt.Figure:
#     """Render a network plot using a fixed layout with optional labels.

#     Parameters
#     ----------
#     G: networkx.Graph
#         The graph to draw.
#     pos: dict
#         Precomputed layout positions for each node.
#     node_size: dict, optional
#         A mapping from node to size. If provided, node sizes are
#         scaled relative to the maximum value in the mapping.
#     node_color: dict, optional
#         A mapping from node to a numeric color value. If provided,
#         colors are mapped through the given colormap.
#     edge_color: dict, optional
#         Mapping from edge (u, v) to numeric color value (e.g. delta Kemeny).
#     cmap: str, optional
#         Name of a Matplotlib colormap to use when mapping numeric values
#         to colors; defaults to ``"viridis"``.
#     highlight_nodes: iterable, optional
#         A collection of nodes to draw with a distinct border.
#     title: str, optional
#         Title for the plot.
#     show_labels: bool, optional
#         If True, draw node labels (usually the node identifiers) on the
#         graph. Font sizes will be scaled with node size to keep
#         proportions.
#     label_dict: dict, optional
#         An optional mapping of nodes to label strings. If None,
#         ``str(node)`` is used for each node.
#     removed_edges: iterable of (u, v), optional
#         Edges to overlay as visually "removed" (drawn as dashed red lines).
#         Useful when you want to keep the overall structure visible while
#         clearly indicating which connections were removed.

#     Returns
#     -------
#     matplotlib.figure.Figure
#         The figure containing the plot.
#     """
#     fig, ax = plt.subplots(figsize=(6, 6))
#     ax.set_axis_off()

#     # Prepare node sizes
#     if node_size is not None:
#         sizes_raw = np.array([node_size.get(n, 1.0) for n in G.nodes()], dtype=float)
#         if sizes_raw.max() > 0:
#             sizes = 300.0 * (sizes_raw / sizes_raw.max())
#         else:
#             sizes = np.full_like(sizes_raw, 100.0)
#     else:
#         sizes_raw = np.ones(G.number_of_nodes(), dtype=float)
#         sizes = np.full(G.number_of_nodes(), 100.0)
#     # Prepare node colors
#     if node_color is not None:
#         values = np.array([node_color.get(n, 0.0) for n in G.nodes()], dtype=float)
#         vmin, vmax = values.min(), values.max()
#         if vmin == vmax:
#             vmin, vmax = 0.0, 1.0
#         colours = values
#     else:
#         colours = np.zeros(G.number_of_nodes())
#         vmin, vmax = 0.0, 1.0
#     # Prepare edge colors (optional heatmap)
#     edges = list(G.edges())
#     edge_values = None
#     edge_vmin = edge_vmax = None

#     if edge_color is not None:
#         edge_values = []
#         for u, v in edges:
#             if (u, v) in edge_color:
#                 edge_values.append(edge_color[(u, v)])
#             elif not G.is_directed() and (v, u) in edge_color:
#                 edge_values.append(edge_color[(v, u)])
#             else:
#                 edge_values.append(0.0)

#         edge_values = np.array(edge_values, dtype=float)

#         max_abs = np.max(np.abs(edge_values))
#         if max_abs > 0:
#             edge_vmin, edge_vmax = -max_abs, max_abs
#         else:
#             edge_vmin, edge_vmax = -1.0, 1.0
#     # Draw nodes
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
#     # Highlight nodes
#     if highlight_nodes:
#         nodelist = list(highlight_nodes)
#         highlight_sizes = [sizes[list(G.nodes()).index(n)] for n in nodelist]
#         highlight_colours = [colours[list(G.nodes()).index(n)] for n in nodelist]
#         # import streamlit as st
#         # st.write(f"highlight_colours ={highlight_colours}")
#         # highlight_colours = ["#ff2fa4" for _ in nodelist]
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
#     if edge_values is not None:
#         nx.draw_networkx_edges(
#             G,
#             pos,
#             edgelist=edges,
#             edge_color=edge_values,
#             edge_cmap=plt.cm.coolwarm,
#             edge_vmin=edge_vmin,
#             edge_vmax=edge_vmax,
#             width=2.5,
#             alpha=0.9,
#             ax=ax,
#         )
#         sm = plt.cm.ScalarMappable(
#         cmap=plt.cm.coolwarm,
#         norm=plt.Normalize(vmin=edge_vmin, vmax=edge_vmax),
#         )
#         sm.set_array([])
#         plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label="Δ Kemeny if edge removed")
#     else:
#         nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)
#     # Overlay removed edges as dashed red (drawn on top)
#     if removed_edges:
#         dashed: List[Tuple[Any, Any]] = list(removed_edges)
#         if not G.is_directed():
#             s = set(dashed)
#             s |= {(v, u) for (u, v) in s}
#             dashed = list(s)
#         nx.draw_networkx_edges(
#             G,
#             pos,
#             edgelist=dashed,
#             ax=ax,
#             edge_color="red",
#             # style="dashed",
#             width=1.4,
#             alpha=1,
#             style=(0, (2, 6)),
#         )

#     # Labels
#     if show_labels:
#         max_raw = sizes_raw.max() if sizes_raw.max() > 0 else 1.0
#         for idx, n in enumerate(G.nodes()):
#             x, y = pos[n]
#             fs = 4 + 3 * (sizes_raw[idx] / max_raw)
#             label = label_dict[n] if (label_dict is not None and n in label_dict) else str(n)
#             ax.text(x, y, label, fontsize=fs, ha="center", va="center", color="white")
#     if title:
#         ax.set_title(title)
#     return fig















# def plot_network(
#     G: nx.Graph,
#     pos: Dict[Any, np.ndarray],
#     *,
#     node_size: Optional[Dict[Any, float]] = None,
#     node_color: Optional[Dict[Any, float]] = None,
#     edge_color: Optional[Dict[Tuple[Any, Any], float]] = None,
#     cmap: str = "viridis",
#     highlight_nodes: Optional[Iterable[Any]] = None,
#     highlight_nodes_selected: Optional[Iterable[Any]] = None,
#     title: Optional[str] = None,
#     show_labels: bool = True,
#     label_dict: Optional[Dict[Any, str]] = None,
#     removed_edges: Optional[Iterable[Tuple[Any, Any]]] = None,
# ) -> plt.Figure:
#     """Render a network plot using a fixed layout with optional labels.

#     Parameters
#     ----------
#     G: networkx.Graph
#         The graph to draw.
#     pos: dict
#         Precomputed layout positions for each node.
#     node_size: dict, optional
#         A mapping from node to size. If provided, node sizes are
#         scaled relative to the maximum value in the mapping.
#     node_color: dict, optional
#         A mapping from node to a numeric color value. If provided,
#         colors are mapped through the given colormap.
#     edge_color: dict, optional
#         Mapping from edge (u, v) to numeric color value (e.g. delta Kemeny).
#     cmap: str, optional
#         Name of a Matplotlib colormap to use when mapping numeric values
#         to colors; defaults to ``"viridis"``.
#     highlight_nodes: iterable, optional
#         Nodes to highlight as "important" (default style: red outline).
#         Typical use: Top N, Bottom N, or any computed highlight group.
#     highlight_nodes_selected: iterable, optional
#         Nodes to highlight as "selected by user" (default style: pink).
#         Typical use: nodes chosen via a multiselect UI.
#     title: str, optional
#         Title for the plot.
#     show_labels: bool, optional
#         If True, draw node labels (usually the node identifiers) on the
#         graph. Font sizes will be scaled with node size to keep
#         proportions.
#     label_dict: dict, optional
#         An optional mapping of nodes to label strings. If None,
#         ``str(node)`` is used for each node.
#     removed_edges: iterable of (u, v), optional
#         Edges to overlay as visually "removed" (drawn as dashed red lines).
#         Useful when you want to keep the overall structure visible while
#         clearly indicating which connections were removed.

#     Returns
#     -------
#     matplotlib.figure.Figure
#         The figure containing the plot.
#     """
#     fig, ax = plt.subplots(figsize=(6, 6))
#     ax.set_axis_off()

#     # Cache node order once to avoid repeated list(G.nodes()) lookups
#     nodes_list = list(G.nodes())
#     idx_map = {n: i for i, n in enumerate(nodes_list)}

#     # Prepare node sizes
#     if node_size is not None:
#         sizes_raw = np.array([node_size.get(n, 1.0) for n in nodes_list], dtype=float)
#         if float(sizes_raw.max()) > 0:
#             sizes = 300.0 * (sizes_raw / float(sizes_raw.max()))
#         else:
#             sizes = np.full_like(sizes_raw, 100.0)
#     else:
#         sizes_raw = np.ones(len(nodes_list), dtype=float)
#         sizes = np.full(len(nodes_list), 100.0)

#     # Prepare node colors (colormap-based, using numeric values)
#     if node_color is not None:
#         values = np.array([node_color.get(n, 0.0) for n in nodes_list], dtype=float)
#         vmin, vmax = float(values.min()), float(values.max())
#         if vmin == vmax:
#             vmin, vmax = 0.0, 1.0
#         colours = values
#     else:
#         colours = np.zeros(len(nodes_list), dtype=float)
#         vmin, vmax = 0.0, 1.0

#     # Prepare edge colors (optional heatmap)
#     edges = list(G.edges())
#     edge_values = None
#     edge_vmin = edge_vmax = None

#     if edge_color is not None:
#         edge_values_list: list[float] = []
#         for u, v in edges:
#             if (u, v) in edge_color:
#                 edge_values_list.append(float(edge_color[(u, v)]))
#             elif not G.is_directed() and (v, u) in edge_color:
#                 edge_values_list.append(float(edge_color[(v, u)]))
#             else:
#                 edge_values_list.append(0.0)

#         edge_values = np.array(edge_values_list, dtype=float)

#         max_abs = float(np.max(np.abs(edge_values)))
#         if max_abs > 0:
#             edge_vmin, edge_vmax = -max_abs, max_abs
#         else:
#             edge_vmin, edge_vmax = -1.0, 1.0

#     # Draw base nodes (colormap face colours)
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

#     # Helper: draw a highlight layer with a distinct outline and optional fixed face colour
#     def _draw_highlight_layer(
#         nodes: Iterable[Any],
#         *,
#         outline_color: str,
#         outline_width: float,
#         fixed_face_color: Optional[str],
#         draw_last: bool = False,
#     ) -> None:
#         """
#         Draw one highlight layer.

#         If fixed_face_color is None, we keep the original colormap-based face colour.
#         If fixed_face_color is a colour string (hex), all highlighted nodes get that face colour.
#         """
#         # Keep only nodes that actually exist in the graph
#         nodelist = [n for n in nodes if n in idx_map]
#         if not nodelist:
#             return

#         # Match sizes to the already drawn nodes
#         highlight_sizes = [float(sizes[idx_map[n]]) for n in nodelist]

#         if fixed_face_color is None:
#             # Preserve original face colours so highlights blend with the heatmap
#             highlight_face = [float(colours[idx_map[n]]) for n in nodelist]
#             nx.draw_networkx_nodes(
#                 G,
#                 pos,
#                 nodelist=nodelist,
#                 node_size=highlight_sizes,
#                 node_color=highlight_face,
#                 cmap=cmap,
#                 vmin=vmin,
#                 vmax=vmax,
#                 edgecolors=outline_color,
#                 linewidths=outline_width,
#                 ax=ax,
#             )
#         else:
#             # Force a constant face colour (e.g. pink) so "selected" nodes stand out
#             nx.draw_networkx_nodes(
#                 G,
#                 pos,
#                 nodelist=nodelist,
#                 node_size=highlight_sizes,
#                 node_color=[fixed_face_color for _ in nodelist],
#                 edgecolors=outline_color,
#                 linewidths=outline_width,
#                 ax=ax,
#             )

#         # Ensure selected highlights appear above other highlight layers if desired
#         if draw_last:
#             # NetworkX returns a PathCollection for nodes; set a higher zorder if available
#             try:
#                 ax.collections[-1].set_zorder(10)
#             except Exception:
#                 pass

#     # Highlight layer 1: computed highlights (Top/Bottom/etc) with red outline
#     if highlight_nodes:
#         _draw_highlight_layer(
#             highlight_nodes,
#             outline_color="red",
#             outline_width=2.0,
#             fixed_face_color=None,  # keep colormap face
#             draw_last=False,
#         )

#     # Highlight layer 2: user-selected nodes with pink outline and pink face
#     # Draw this after the red layer so pink wins if a node is in both sets.
#     if highlight_nodes_selected:
#         _draw_highlight_layer(
#             highlight_nodes_selected,
#             outline_color="#ff2fa4",
#             outline_width=2.6,
#             fixed_face_color="#ff2fa4",
#             draw_last=True,
#         )

#     # Draw edges (optionally with edge heatmap)
#     if edge_values is not None:
#         nx.draw_networkx_edges(
#             G,
#             pos,
#             edgelist=edges,
#             edge_color=edge_values,
#             edge_cmap=plt.cm.coolwarm,
#             edge_vmin=edge_vmin,
#             edge_vmax=edge_vmax,
#             width=2.5,
#             alpha=0.9,
#             ax=ax,
#         )
#         sm = plt.cm.ScalarMappable(
#             cmap=plt.cm.coolwarm,
#             norm=plt.Normalize(vmin=edge_vmin, vmax=edge_vmax),
#         )
#         sm.set_array([])
#         plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label="Δ Kemeny if edge removed")
#     else:
#         nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)

#     # Overlay removed edges as dashed red (drawn on top)
#     if removed_edges:
#         dashed: List[Tuple[Any, Any]] = list(removed_edges)
#         if not G.is_directed():
#             s = set(dashed)
#             s |= {(v, u) for (u, v) in s}
#             dashed = list(s)

#         nx.draw_networkx_edges(
#             G,
#             pos,
#             edgelist=dashed,
#             ax=ax,
#             edge_color="red",
#             width=1.4,
#             alpha=1,
#             style=(0, (2, 6)),
#         )

#     # Labels
#     if show_labels:
#         max_raw = float(sizes_raw.max()) if float(sizes_raw.max()) > 0 else 1.0
#         for idx, n in enumerate(nodes_list):
#             x, y = pos[n]
#             fs = 4 + 3 * (float(sizes_raw[idx]) / max_raw)
#             label = label_dict[n] if (label_dict is not None and n in label_dict) else str(n)
#             ax.text(x, y, label, fontsize=float(fs), ha="center", va="center", color="white")

#     if title:
#         ax.set_title(title)

#     return fig



def plot_network(
    G: nx.Graph,
    pos: Dict[Any, np.ndarray],
    *,
    node_size: Optional[Dict[Any, float]] = None,
    node_color: Optional[Dict[Any, float]] = None,
    edge_color: Optional[Dict[Tuple[Any, Any], float]] = None,
    cmap: str = "viridis",
    highlight_nodes: Optional[Iterable[Any]] = None,
    highlight_nodes_selected: Optional[Iterable[Any]] = None,
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
    edge_color: dict, optional
        Mapping from edge (u, v) to numeric color value (e.g. delta Kemeny).
    cmap: str, optional
        Name of a Matplotlib colormap to use when mapping numeric values
        to colors; defaults to "viridis".
    highlight_nodes: iterable, optional
        Nodes to highlight as "important" using a red outline.
        Typical use: Top N, Bottom N, or any computed highlight group.
    highlight_nodes_selected: iterable, optional
        Nodes selected by the user (for example via a multiselect).
        These nodes are highlighted with a pink outline only.
        The node face colour remains colormap-based.
    title: str, optional
        Title for the plot.
    show_labels: bool, optional
        If True, draw node labels (usually the node identifiers) on the graph.
        Font sizes will be scaled with node size to keep proportions.
    label_dict: dict, optional
        Optional mapping of nodes to label strings. If None, str(node) is used.
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

    # Cache node order once to avoid repeated list(G.nodes()) lookups.
    # This also ensures consistent indexing across sizes, colours, and labels.
    nodes_list = list(G.nodes())
    idx_map = {n: i for i, n in enumerate(nodes_list)}

    # -------------------------
    # Node sizes (scaled)
    # -------------------------
    if node_size is not None:
        sizes_raw = np.array([node_size.get(n, 1.0) for n in nodes_list], dtype=float)
        if float(sizes_raw.max()) > 0:
            sizes = 300.0 * (sizes_raw / float(sizes_raw.max()))
        else:
            sizes = np.full_like(sizes_raw, 100.0)
    else:
        sizes_raw = np.ones(len(nodes_list), dtype=float)
        sizes = np.full(len(nodes_list), 100.0)

    # -------------------------
    # Node colours (numeric values mapped through a colormap)
    # -------------------------
    if node_color is not None:
        values = np.array([node_color.get(n, 0.0) for n in nodes_list], dtype=float)
        vmin, vmax = float(values.min()), float(values.max())
        if vmin == vmax:
            # Avoid a degenerate colormap range
            vmin, vmax = 0.0, 1.0
        colours = values
    else:
        colours = np.zeros(len(nodes_list), dtype=float)
        vmin, vmax = 0.0, 1.0

    # -------------------------
    # Optional edge heatmap colours
    # -------------------------
    edges = list(G.edges())
    edge_values = None
    edge_vmin = edge_vmax = None

    if edge_color is not None:
        edge_values_list: list[float] = []
        for u, v in edges:
            if (u, v) in edge_color:
                edge_values_list.append(float(edge_color[(u, v)]))
            elif not G.is_directed() and (v, u) in edge_color:
                edge_values_list.append(float(edge_color[(v, u)]))
            else:
                edge_values_list.append(0.0)

        edge_values = np.array(edge_values_list, dtype=float)

        max_abs = float(np.max(np.abs(edge_values)))
        if max_abs > 0:
            edge_vmin, edge_vmax = -max_abs, max_abs
        else:
            edge_vmin, edge_vmax = -1.0, 1.0

    # -------------------------
    # Draw base nodes (face colours from colormap)
    # -------------------------
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

    # -------------------------
    # Highlight layers (outlines only)
    # -------------------------
    def _draw_highlight_layer(
        nodes: Iterable[Any],
        *,
        outline_color: str,
        outline_width: float,
        draw_last: bool = False,
    ) -> None:
        """Draw a highlight overlay for a set of nodes.

        This overlay:
        - Keeps the original face colour (colormap-based)
        - Adds a coloured outline using edgecolors + linewidths

        draw_last can be used to ensure the overlay appears above previously
        drawn overlays if nodes are in multiple highlight sets.
        """
        nodelist = [n for n in nodes if n in idx_map]
        if not nodelist:
            return

        highlight_sizes = [float(sizes[idx_map[n]]) for n in nodelist]
        highlight_face = [float(colours[idx_map[n]]) for n in nodelist]

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodelist,
            node_size=highlight_sizes,
            node_color=highlight_face,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            edgecolors=outline_color,
            linewidths=outline_width,
            ax=ax,
        )

        # Try to push this highlight layer above earlier collections
        if draw_last:
            try:
                ax.collections[-1].set_zorder(10)
            except Exception:
                pass

    # Layer 1: computed highlights (Top N, Bottom N, etc) with red outline
    if highlight_nodes:
        _draw_highlight_layer(
            highlight_nodes,
            outline_color="red",
            outline_width=2.0,
            draw_last=False,
        )

    # Layer 2: user-selected nodes with pink outline only (no face recolour)
    # Draw after red so pink wins if a node is in both sets.
    if highlight_nodes_selected:
        _draw_highlight_layer(
            highlight_nodes_selected,
            outline_color="#ff2fa4",
            outline_width=2.6,
            draw_last=True,
        )

    # -------------------------
    # Draw edges (either heatmap or default)
    # -------------------------
    if edge_values is not None:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edges,
            edge_color=edge_values,
            edge_cmap=plt.cm.coolwarm,
            edge_vmin=edge_vmin,
            edge_vmax=edge_vmax,
            width=2.5,
            alpha=0.9,
            ax=ax,
        )

        # Add colorbar for edge heatmap
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.coolwarm,
            norm=plt.Normalize(vmin=edge_vmin, vmax=edge_vmax),
        )
        sm.set_array([])
        plt.colorbar(
            sm,
            ax=ax,
            fraction=0.046,
            pad=0.04,
            label="Δ Kemeny if edge removed",
        )
    else:
        nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)

    # -------------------------
    # Overlay removed edges (always drawn on top)
    # -------------------------
    if removed_edges:
        dashed: List[Tuple[Any, Any]] = list(removed_edges)

        # If undirected, draw both (u, v) and (v, u) to avoid missing overlays
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
            width=1.4,
            alpha=1,
            style=(0, (2, 6)),
        )

    # -------------------------
    # Labels
    # -------------------------
    if show_labels:
        max_raw = float(sizes_raw.max()) if float(sizes_raw.max()) > 0 else 1.0
        for idx, n in enumerate(nodes_list):
            x, y = pos[n]
            fs = 4 + 3 * (float(sizes_raw[idx]) / max_raw)
            label = label_dict[n] if (label_dict is not None and n in label_dict) else str(n)
            ax.text(
                x,
                y,
                label,
                fontsize=float(fs),
                ha="center",
                va="center",
                color="white",
            )

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
    
