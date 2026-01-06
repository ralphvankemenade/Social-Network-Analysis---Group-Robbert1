"""Streamlit page: Arrest optimisation model."""

import streamlit as st
import pandas as pd

from dss.ui.state import init_state, get_state, set_state
from dss.analytics.communities import compute_communities
from dss.analytics.centrality import compute_centralities, combine_centralities
from dss.analytics.arrest_optimization import arrest_assignment
from dss.ui.components import display_network


def page() -> None:
    st.set_page_config(page_title="Arrest Optimisation", layout="wide")
    st.title("Arrest Optimisation")
    init_state()
    G = get_state("graph")
    if G is None:
        st.info("No graph loaded.  Please upload a `.mtx` file on the Upload page.")
        return
    # Sidebar parameters
    st.sidebar.header("Arrest optimisation parameters")
    comm_method = st.sidebar.selectbox("Community method for regret weights", ["louvain", "girvan_newman", "spectral"], index=0)
    alpha = st.sidebar.slider("Regret strength (alpha)", 0.0, 5.0, 1.0, 0.1)
    beta = st.sidebar.slider("Penalty strength (beta)", 0.0, 5.0, 1.0, 0.1)
    centrality_metric = st.sidebar.selectbox("Centrality for regret", ["None", "degree", "combined"], index=2)
    if st.sidebar.button("Compute arrest assignment") or get_state("arrest_result") is None:
        # Compute community labels
        if get_state("community_results").get(comm_method) is None:
            comm_result = compute_communities(G, method=comm_method, k=2)
            get_state("community_results")[comm_method] = comm_result
        comm_result = get_state("community_results")[comm_method]
        communities = comm_result.labels
        # Centrality scores
        if centrality_metric == "None":
            centrality_scores = None
        else:
            if get_state("centrality_result") is None:
                from dss.analytics.centrality import compute_centrality_result
                result = compute_centrality_result(G)
                set_state("centrality_result", result)
            centrality_table = get_state("centrality_result").table
            if centrality_metric == "combined":
                combined = combine_centralities(centrality_table)
                centrality_scores = combined
            else:
                centrality_scores = centrality_table[centrality_metric]
        # Compute assignment
        arrest_result = arrest_assignment(G, communities, centrality_scores, alpha=alpha, beta=beta)
        set_state("arrest_result", arrest_result)
    arrest_result = get_state("arrest_result")
    if arrest_result is not None:
        st.subheader("Optimisation results")
        st.write(f"Objective value: {arrest_result.objective:.3f}")
        st.write(f"Cross‑department edges: {arrest_result.cut_edges}")
        st.write(f"Estimated effective arrests: {arrest_result.effective_arrests:.3f}")
        # Display network coloured by department
        dept_colors = {node: arrest_result.assignment[node] for node in G.nodes()}
        display_network(G, node_color=dept_colors, title="Department assignment (0/1)")
        # Show list of risky edges
        if arrest_result.risk_edges:
            df_risk = pd.DataFrame(arrest_result.risk_edges, columns=["u", "v"])
            st.subheader("Edges across departments (risk)")
            st.dataframe(df_risk)
    else:
        st.info("No optimisation result available yet.  Adjust parameters and press the button to compute.")


if __name__ == "__main__":
    page()