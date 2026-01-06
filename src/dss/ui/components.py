"""Reusable UI components for the DSS.

This module wraps common Streamlit UI patterns into functions to keep
the page code concise.  Components include network visualisation,
tables, metrics cards and charts.
"""

from typing import Any, Iterable, Dict, Optional
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils.plotting import plot_network_interactive
from ..graph.layouts import compute_layout


def display_network(
    G,
    node_size: Optional[Dict[Any, float]] = None,
    node_color: Optional[Dict[Any, float]] = None,
    highlight: Optional[Iterable[Any]] = None,
    hover: Optional[Dict[Any, str]] = None,
    title: Optional[str] = None,
    label_dict: Optional[Dict[Any, str]] = None,
    symbol_dict: Optional[Dict[Any, str]] = None,
) -> None:
    """Render an interactive network graph using Streamlit.

    This function wraps ``plot_network_interactive`` to create a Plotly
    figure and then embeds it in the Streamlit app using
    ``st.plotly_chart``.  If no graph is loaded, a message is shown
    instead.

    Parameters
    ----------
    G: networkx.Graph
        The graph to render.
    node_size: dict, optional
        Mapping from node to size.  Values are scaled internally.
    node_color: dict, optional
        Mapping from node to colour value.  Values are mapped to a
        colour scale.
    highlight: iterable, optional
        Nodes to highlight with a thick red border.
    hover: dict, optional
        Mapping from node to hover text.  If provided, this text
        appears when hovering over a node.
    title: str, optional
        Title for the plot.
    label_dict: dict, optional
        Custom labels for nodes; defaults to node identifiers.
    """
    if G is None or G.number_of_nodes() == 0:
        st.info("No graph loaded.")
        return
    # Compute a deterministic layout.  In the absence of caching, this
    # function computes the layout anew on every call.  For small
    # networks (<100 nodes) the overhead is negligible.
    pos = compute_layout(G)
    fig = plot_network_interactive(
        G,
        pos,
        node_size=node_size,
        node_color=node_color,
        highlight_nodes=highlight,
        label_dict=label_dict,
        hover_dict=hover,
        symbol_dict=symbol_dict,
        title=title,
    )
    st.plotly_chart(fig, use_container_width=True)


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
