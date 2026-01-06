"""Graph plotting utilities.

This module centralises the creation of network plots for the DSS.  Two
separate plotting backends are provided:

* A static Matplotlib implementation (`plot_network`) which is
  retained for completeness and non‑interactive contexts.
* A Plotly implementation (`plot_network_interactive`) which
  produces an interactive figure suitable for embedding in a
  Streamlit dashboard.  Nodes display their identifiers directly on
  the canvas and expose additional information via hover text.  Nodes
  can also be highlighted by modifying their marker borders.  This
  function is used throughout the updated DSS to give users an
  interactive experience when exploring their networks.

Using Plotly avoids relying on third‑party components and works with
packages already available in the environment.  Unlike Matplotlib
figures, Plotly charts support panning, zooming and tooltips out of
the box, which addresses the user’s request to be able to inspect
individual nodes interactively.  See the page implementations under
``src/dss/pages`` for examples of how to integrate the returned
figures into Streamlit via ``st.plotly_chart``.
"""

from typing import Dict, Iterable, Optional, Any, List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import plotly.graph_objects as go



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
) -> plt.Figure:
    """Render a network plot using a fixed layout with optional labels.

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
        to colours; defaults to ``"viridis"``.
    highlight_nodes: iterable, optional
        A collection of nodes to draw with a distinct border.
    title: str, optional
        Title for the plot.
    show_labels: bool, optional
        If True, draw node labels (usually the node identifiers) on the
        graph.  Font sizes will be scaled with node size to keep
        proportions.
    label_dict: dict, optional
        An optional mapping of nodes to label strings.  If None,
        ``str(node)`` is used for each node.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plot.
    """

    # Create a square figure for consistent aspect ratio
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_axis_off()
    # Prepare node sizes.  Use the raw values to compute scaling and
    # later derive font sizes.  If not provided, default to uniform size.
    if node_size is not None:
        sizes_raw = np.array([node_size.get(n, 1.0) for n in G.nodes()], dtype=float)
        if sizes_raw.max() > 0:
            sizes = 300.0 * (sizes_raw / sizes_raw.max())
        else:
            sizes = np.full_like(sizes_raw, 100.0)
    else:
        sizes_raw = np.ones(G.number_of_nodes(), dtype=float)
        sizes = np.full(G.number_of_nodes(), 100.0)
    # Prepare node colours.  Normalize values to a colour scale.  If
    # uniform or unspecified, assign a constant zero value to each node.
    if node_color is not None:
        values = np.array([node_color.get(n, 0.0) for n in G.nodes()], dtype=float)
        vmin, vmax = values.min(), values.max()
        if vmin == vmax:
            vmin, vmax = 0.0, 1.0
        colours = values
    else:
        colours = np.zeros(G.number_of_nodes())
        vmin, vmax = 0.0, 1.0
    # Draw nodes with sizes and colours
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
    # Highlight specified nodes by re‑drawing them with a red border
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
            linewidths=2.0,
            ax=ax,
        )
    # Draw edges with light transparency
    nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)
    # Draw labels if requested.  Font sizes are scaled between 6 and 12
    # based on the relative node size.  If a custom ``label_dict`` is
    # provided, its values are used; otherwise use the node identifier.
    if show_labels:
        max_raw = sizes_raw.max() if sizes_raw.max() > 0 else 1.0
        for idx, n in enumerate(G.nodes()):
            x, y = pos[n]
            # Scale font size: base of 6 plus up to 6 additional points
            fs = 6 + 6 * (sizes_raw[idx] / max_raw)
            label = label_dict[n] if (label_dict is not None and n in label_dict) else str(n)
            ax.text(x, y, label, fontsize=fs, ha='center', va='center', color='black')
    # Set title if provided
    if title:
        ax.set_title(title)
    return fig


def plot_network_interactive(
    G: nx.Graph,
    pos: Dict[Any, np.ndarray],
    *,
    node_size: Optional[Dict[Any, float]] = None,
    node_color: Optional[Dict[Any, float]] = None,
    highlight_nodes: Optional[Iterable[Any]] = None,
    label_dict: Optional[Dict[Any, str]] = None,
    hover_dict: Optional[Dict[Any, str]] = None,
    symbol_dict: Optional[Dict[Any, str]] = None,
    title: Optional[str] = None,
) -> go.Figure:
    """Render an interactive network plot using Plotly.

    Parameters
    ----------
    G: networkx.Graph
        The graph to draw.
    pos: dict
        Precomputed layout positions for each node.
    node_size: dict, optional
        Mapping from node to size.  Values will be normalised to a
        reasonable marker size range.  If omitted, all nodes are
        displayed with the same size.
    node_color: dict, optional
        Mapping from node to a numeric colour value.  Values are
        passed directly to the marker colour array.  If omitted,
        nodes are coloured uniformly.
    highlight_nodes: iterable, optional
        Nodes to highlight with a thicker border.  This is useful for
        emphasising selected nodes, such as those removed in the
        Kemeny analysis or the top/bottom centrality nodes.
    label_dict: dict, optional
        Optional mapping of nodes to label strings.  Defaults to
        ``str(node)``.  Labels are displayed on the node markers.
    hover_dict: dict, optional
        Optional mapping of nodes to additional hover text.  If
        provided, these strings will appear in the hover tooltip for
        each node.  This can be used to display centrality values,
        role or community assignments, etc.
    title: str, optional
        Title for the plot.

    Returns
    -------
    plotly.graph_objects.Figure
        The Plotly figure ready to be passed to ``st.plotly_chart``.
    """
    # Extract positions
    nodes = list(G.nodes())
    x_vals: List[float] = []
    y_vals: List[float] = []
    for n in nodes:
        x_vals.append(pos[n][0])
        y_vals.append(pos[n][1])
    # Scale node sizes to a reasonable range
    if node_size is not None:
        raw_sizes = np.array([node_size.get(n, 1.0) for n in nodes], dtype=float)
        max_raw = raw_sizes.max() if raw_sizes.size > 0 else 1.0
        if max_raw == 0:
            max_raw = 1.0
        sizes = 20.0 + 40.0 * (raw_sizes / max_raw)  # size between 20 and 60
    else:
        sizes = np.full(len(nodes), 30.0)
    # Colours
    if node_color is not None:
        colours = [node_color.get(n, 0.0) for n in nodes]
    else:
        colours = [0.0 for _ in nodes]
    # Determine border widths for highlighted nodes
    if highlight_nodes:
        hl_set = set(highlight_nodes)
        line_widths = [4.0 if n in hl_set else 1.0 for n in nodes]
        line_colours = ['red' if n in hl_set else '#333333' for n in nodes]
    else:
        line_widths = [1.0 for _ in nodes]
        line_colours = ['#333333' for _ in nodes]
    # Prepare hover text
    if hover_dict:
        hovertext = [hover_dict.get(n, '') for n in nodes]
    else:
        hovertext = [str(n) for n in nodes]
    # Labels to display on the node
    if label_dict:
        labels = [label_dict.get(n, str(n)) for n in nodes]
    else:
        labels = [str(n) for n in nodes]
    # Create edge traces.  Each edge is a separate line segment.  To
    # avoid drawing separate traces for every edge (which can be
    # inefficient), we gather all edges into a single scatter trace
    # consisting of many line segments.  For each edge we insert a None
    # separator to break the path.
    edge_x: List[float] = []
    edge_y: List[float] = []
    for (u, v) in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1.0, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    # Determine marker symbols.  Plotly allows a list of symbols per marker.
    if symbol_dict:
        symbols = [symbol_dict.get(n, 'circle') for n in nodes]
    else:
        symbols = ['circle' for _ in nodes]
    node_trace = go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='markers+text',
        text=labels,
        textposition='middle center',
        textfont=dict(color='black', size=10),
        hovertext=hovertext,
        hoverinfo='text',
        marker=dict(
            symbol=symbols,
            showscale=True,
            colorscale='Viridis',
            color=colours,
            size=sizes,
            line=dict(color=line_colours, width=line_widths),
            colorbar=dict(thickness=15, title='Value', xanchor='left', titleside='right'),
        )
    )
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
    )
    if title:
        fig.update_layout(title=title)
    # Fix the axis to equal aspect ratio and hide axes
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
    return fig


if __name__ == "__main__":
    # Simple demonstration
    import networkx as nx
    G = nx.cycle_graph(5)
    pos = nx.spring_layout(G, seed=42)
    fig = plot_network(G, pos, title="Cycle graph")
    fig.savefig("_demo_plot.png")
    print("Demo plot saved to _demo_plot.png")