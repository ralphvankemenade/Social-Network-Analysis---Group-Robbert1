"""Streamlit page: Centrality analysis.

This page computes a suite of centrality measures, displays them in a
table, allows the user to highlight top and bottom nodes, and combines
metrics via a weighted sum or Borda count.  Users can download the
centrality table as CSV.
"""

import streamlit as st
import pandas as pd
from dss.ui.state import init_state, get_state, set_state
from dss.analytics.centrality import compute_centrality_result, combine_centralities, borda_count
from dss.ui.components import display_network


def page() -> None:
    st.set_page_config(page_title="Centrality Analysis", layout="wide")
    st.title("Centrality Analysis")
    init_state()
    G = get_state("graph")
    if G is None:
        st.info("No graph loaded.  Please upload a `.mtx` file on the Upload page.")
        return
    # Compute or retrieve centrality result
    if get_state("centrality_result") is None:
        result = compute_centrality_result(G)
        set_state("centrality_result", result)
    else:
        result = get_state("centrality_result")
    df = result.table
    combined_scores = result.combined_scores
    ranks = result.ranks
    st.subheader("Centrality measures")
    # Sidebar for weighting scheme
    st.sidebar.header("Weighting scheme")
    weight_inputs = {}
    for col in df.columns:
        weight_inputs[col] = st.sidebar.slider(f"Weight for {col}", 0.0, 1.0, 1.0, 0.1)
    # Choose aggregation method
    agg_method = st.sidebar.radio("Aggregation method", ["Weighted sum", "Borda count"], index=0)
    if agg_method == "Weighted sum":
        combined = combine_centralities(df, weights=weight_inputs)
    else:
        combined = borda_count(df)
    # Display centrality table
    st.dataframe(df.assign(combined=combined).sort_values("combined", ascending=False))
    # Download as CSV
    csv_data = df.assign(combined=combined).to_csv().encode("utf-8")
    st.download_button("Download centrality data as CSV", csv_data, file_name="centrality.csv", mime="text/csv")
    # Highlight controls and node selection
    st.sidebar.header("Highlight and select nodes")
    max_n = min(20, len(df))
    top_n = st.sidebar.slider("Top N", 1, max_n, min(5, max_n))
    highlight_top = st.sidebar.checkbox("Highlight top N", value=True)
    highlight_bottom = st.sidebar.checkbox("Highlight bottom N", value=False)
    # Node selection for detailed view
    selected_nodes = st.sidebar.multiselect(
        "Select nodes to inspect", options=list(G.nodes()), default=[]
    )
    # Determine highlight nodes: top/bottom plus selected
    highlight_nodes = []
    if highlight_top:
        highlight_nodes += combined.nlargest(top_n).index.tolist()
    if highlight_bottom:
        highlight_nodes += combined.nsmallest(top_n).index.tolist()
    # Always include explicitly selected nodes in highlight list
    highlight_nodes += [n for n in selected_nodes if n not in highlight_nodes]
    st.subheader("Network with node size by aggregated centrality")
    # Map node sizes and colours.  Use the combined score both for
    # sizing and colouring to emphasise highly central nodes.  Build
    # hover text containing all centrality values and the combined
    # score.
    size_map = combined.to_dict()
    color_map = combined.to_dict()
    hover_dict = {}
    for node in G.nodes():
        # Construct a HTML string with each centrality measure on a new line
        parts = []
        for col in df.columns:
            parts.append(f"{col}: {df.at[node, col]:.4f}")
        parts.append(f"Combined: {combined.at[node]:.4f}")
        hover_dict[node] = "<br>".join(parts)
    display_network(
        G,
        node_size=size_map,
        node_color=color_map,
        highlight=highlight_nodes,
        hover=hover_dict,
        title="Centrality‑scaled network",
        label_dict=None,
    )
    # Show information for selected nodes
    if selected_nodes:
        st.subheader("Selected node details")
        info_df = df.loc[selected_nodes].copy()
        info_df["combined"] = combined.loc[selected_nodes]
        st.dataframe(info_df)


if __name__ == "__main__":
    page()