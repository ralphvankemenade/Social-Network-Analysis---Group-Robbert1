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
    # Highlight controls
    st.sidebar.header("Highlight nodes")
    max_n = min(20, len(df))
    top_n = st.sidebar.slider("Top N", 1, max_n, min(5, max_n))
    highlight_top = st.sidebar.checkbox("Highlight top N", value=True)
    highlight_bottom = st.sidebar.checkbox("Highlight bottom N", value=False)
    # Determine highlight nodes
    highlight_nodes = []
    if highlight_top:
        highlight_nodes += combined.nlargest(top_n).index.tolist()
    if highlight_bottom:
        highlight_nodes += combined.nsmallest(top_n).index.tolist()
    st.subheader("Network with node size by aggregated centrality")
    # Map node sizes
    size_map = combined.to_dict()
    display_network(G, node_size=size_map, node_color=combined.to_dict(), highlight=highlight_nodes, title="Centrality‑scaled network")


if __name__ == "__main__":
    page()