"""Reusable UI components for the DSS.

This module wraps common Streamlit UI patterns into functions to keep
the page code concise.  Components include network visualisation,
tables, metrics cards and charts.
"""

from typing import Any, Dict, Iterable, Optional, Tuple
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils.plotting import plot_network
from ..graph.layouts import compute_layout


# def display_network(
#     G,
#     node_size: Optional[Dict[Any, float]] = None,
#     node_color: Optional[Dict[Any, float]] = None,
#     edge_color: Optional[Dict[Tuple[Any, Any], float]] = None,
#     highlight: Optional[Iterable[Any]] = None,
#     title: Optional[str] = None,
#     show_labels: bool = True,
#     label_dict: Optional[Dict[Any, str]] = None,
#     removed_edges: Optional[Iterable[Tuple[Any, Any]]] = None,
# ) -> None:
#     """Render a network graph using Streamlit.

#     Parameters
#     ----------
#     G: networkx.Graph
#         The graph to render.
#     node_size: dict, optional
#         Mapping from node to size. Values are scaled internally.
#     node_color: dict, optional
#         Mapping from node to color value. Values are mapped to a color scale.
#     highlight: iterable, optional
#         Nodes to highlight with a red border.
#     title: str, optional
#         Title for the plot.
#     show_labels: bool, optional
#         If True, draw labels on nodes. Font sizes adjust automatically.
#     label_dict: dict, optional
#         Custom labels for nodes; defaults to node identifiers.
#     removed_edges: iterable of (u, v), optional
#         Edges to overlay as visually "removed" (drawn as dashed red lines).
#         Useful when you want to keep the overall structure visible while
#         clearly indicating which connections were removed.
#     """
#     if G is None or G.number_of_nodes() == 0:
#         st.info("No graph loaded.")
#         return
#     # Compute a deterministic layout. For interactive use the layout is
#     # cached across calls, but caching is disabled in this simplified version.
#     pos = compute_layout(G)
#     fig = plot_network(
#         G,
#         pos,
#         node_size=node_size,
#         node_color=node_color,
#         edge_color=edge_color,
#         highlight_nodes=highlight,
#         title=title,
#         show_labels=show_labels,
#         label_dict=label_dict,
#         removed_edges=removed_edges,
#     )
#     st.pyplot(fig)


def display_network(
    G,
    node_size: Optional[Dict[Any, float]] = None,
    node_color: Optional[Dict[Any, float]] = None,
    edge_color: Optional[Dict[Tuple[Any, Any], float]] = None,
    highlight_top: Optional[Iterable[Any]] = None,
    highlight_selected: Optional[Iterable[Any]] = None,
    title: Optional[str] = None,
    show_labels: bool = True,
    label_dict: Optional[Dict[Any, str]] = None,
    removed_edges: Optional[Iterable[Tuple[Any, Any]]] = None,
) -> None:
    """Render a network graph using Streamlit.

    Parameters
    ----------
    G: networkx.Graph
        The graph to render.
    node_size: dict, optional
        Mapping from node to size. Values are scaled internally.
    node_color: dict, optional
        Mapping from node to color value. Values are mapped to a color scale.
    edge_color: dict, optional
        Mapping from edge (u, v) to numeric color value (e.g. delta Kemeny).
    highlight_top: iterable, optional
        Nodes to highlight as "important" (for example Top N or Bottom N).
        These will be rendered using the default highlight style (red outline).
    highlight_selected: iterable, optional
        Nodes selected by the user (for example via a multiselect).
        These will be rendered using the selected highlight style (pink).
    title: str, optional
        Title for the plot.
    show_labels: bool, optional
        If True, draw labels on nodes. Font sizes adjust automatically.
    label_dict: dict, optional
        Custom labels for nodes; defaults to node identifiers.
    removed_edges: iterable of (u, v), optional
        Edges to overlay as visually "removed" (drawn as dashed red lines).
        Useful when you want to keep the overall structure visible while
        clearly indicating which connections were removed.
    """
    if G is None or G.number_of_nodes() == 0:
        st.info("No graph loaded.")
        return

    # Compute a deterministic layout. For interactive use the layout is typically cached
    # across calls (so node positions remain stable), but caching may be handled elsewhere.
    pos = compute_layout(G)

    # Delegate all drawing decisions to plot_network, including highlight styling.
    fig = plot_network(
        G,
        pos,
        node_size=node_size,
        node_color=node_color,
        edge_color=edge_color,
        highlight_nodes=highlight_top,
        highlight_nodes_selected=highlight_selected,
        title=title,
        show_labels=show_labels,
        label_dict=label_dict,
        removed_edges=removed_edges,
    )

    st.pyplot(fig)




def display_table(df: pd.DataFrame, caption: Optional[str] = None) -> None:
    """Display a DataFrame in Streamlit with a caption."""
    st.dataframe(df)
    if caption:
        st.caption(caption)


def display_heatmap(similarity: Any, nodes: Iterable[Any], caption: Optional[str] = None) -> None:
    """Display a similarity matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(similarity, ax=ax, cmap="viridis")
    ax.set_title(caption or "Similarity matrix")
    st.pyplot(fig)


def display_histogram(data: Iterable[float], title: str, xlabel: str) -> None:
    """Display a histogram for robustness scores."""
    fig, ax = plt.subplots()
    ax.hist(list(data), bins=20, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    st.pyplot(fig)


def display_boxplot(data: Iterable[float], title: str, ylabel: str) -> None:
    """Display a box plot for robustness scores."""
    fig, ax = plt.subplots()
    ax.boxplot(list(data))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    st.pyplot(fig)
