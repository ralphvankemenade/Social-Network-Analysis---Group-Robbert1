"""Streamlit page: Community detection and robustness analysis."""

import streamlit as st
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from dss.ui.state import init_state, get_state, set_state
from dss.analytics.communities import compute_communities
from dss.analytics.robustness import perturbation_test
from dss.ui.components import display_network, display_histogram, display_boxplot
from dss.analytics.roles import compute_roles


def page() -> None:
    st.set_page_config(page_title="Communities & Robustness", layout="wide")
    st.title("Community Detection and Robustness")
    init_state()
    G = get_state("graph")
    if G is None:
        st.info("No graph loaded.  Please upload a `.mtx` file on the Upload page.")
        return
    # Sidebar: choose method and parameters
    st.sidebar.header("Community detection parameters")
    method = st.sidebar.selectbox("Method", ["louvain", "girvan_newman", "spectral"], index=0)
    if method in {"girvan_newman", "spectral"}:
        k = st.sidebar.slider("Number of clusters (k)", 2, max(2, int(G.number_of_nodes() / 2)), 2)
    else:
        k = None
    # Compute communities
    if get_state("community_results").get(method) is None:
        comm_result = compute_communities(G, method=method, k=k)
        get_state("community_results")[method] = comm_result
    comm_result = get_state("community_results")[method]
    # Display summary
    st.subheader("Community summary")
    st.write(f"Modularity Q: {comm_result.modularity:.3f}")
    st.dataframe(comm_result.summary)
    # Network plot coloured by communities with node selection
    community_colors = {node: comm_result.labels[node] for node in G.nodes()}
    st.subheader("Network coloured by communities")
    # Allow user to select nodes for inspection
    st.sidebar.subheader("Select nodes to inspect")
    selected_nodes = st.sidebar.multiselect(
        "Nodes", options=list(G.nodes()), default=[]
    )
    highlight_nodes = selected_nodes
    # Build hover text: include community and centrality measures (and role
    # if available)
    from dss.analytics.centrality import compute_centralities
    centralities_all = compute_centralities(G)
    hover_dict = {}
    role_result = get_state("role_result")
    for node in G.nodes():
        parts = [f"Community: {comm_result.labels[node]}"]
        if role_result is not None:
            parts.append(f"Role: {role_result.labels.get(node, '')}")
        for col in centralities_all.columns:
            parts.append(f"{col}: {centralities_all.at[node, col]:.4f}")
        hover_dict[node] = "<br>".join(parts)
    display_network(
        G,
        node_color=community_colors,
        highlight=highlight_nodes,
        hover=hover_dict,
        title=f"Communities ({method})",
        label_dict=None,
    )
    # Show details for selected nodes
    if selected_nodes:
        st.subheader("Selected node details")
        import pandas as pd  # imported here to avoid top-level import issues
        from dss.analytics.centrality import compute_centralities
        centralities = compute_centralities(G)
        df_details = centralities.loc[selected_nodes].copy()
        df_details["community"] = [comm_result.labels[n] for n in selected_nodes]
        # Also show role cluster if available
        if get_state("role_result") is not None:
            role_result = get_state("role_result")
            df_details["role"] = [role_result.labels[n] for n in selected_nodes]
        st.dataframe(df_details)
    # Robustness analysis
    st.subheader("Robustness analysis")
    runs = st.sidebar.slider("Number of perturbation runs", 10, 100, 50)
    p = st.sidebar.slider("Fraction of edges to remove", 0.01, 0.30, 0.05, 0.01)
    if st.sidebar.button("Run robustness test"):
        robustness_result = perturbation_test(G, method=method, p=p, runs=runs, k=(k or 2))
        set_state("robustness_result", robustness_result)
    robustness_result = get_state("robustness_result")
    if robustness_result is not None:
        st.write(f"Average ARI across runs: {sum(robustness_result.ari_scores) / len(robustness_result.ari_scores):.3f}")
        st.write(f"Average modularity drop: {sum(robustness_result.modularity_drops) / len(robustness_result.modularity_drops):.3f}")
        display_histogram(robustness_result.ari_scores, title="ARI distribution", xlabel="ARI")
        display_boxplot(robustness_result.modularity_drops, title="Modularity drop distribution", ylabel="ΔQ")
    # Compare to roles
    st.subheader("Comparison with role clustering")
    # Compute role result if not already
    if get_state("role_result") is None:
        from dss.analytics.centrality import compute_centralities
        centrality_table = compute_centralities(G)
        role_result = compute_roles(G, centrality_table=centrality_table)
        set_state("role_result", role_result)
    role_result = get_state("role_result")
    role_labels_list = [role_result.labels[node] for node in G.nodes()]
    comm_labels_list = [comm_result.labels[node] for node in G.nodes()]
    ari = adjusted_rand_score(role_labels_list, comm_labels_list)
    nmi = normalized_mutual_info_score(role_labels_list, comm_labels_list)
    st.write(f"Adjusted Rand Index between roles and communities: {ari:.3f}")
    st.write(f"Normalized Mutual Information: {nmi:.3f}")
    confusion = pd.crosstab(pd.Series(role_labels_list, name="role"), pd.Series(comm_labels_list, name="community"))
    st.dataframe(confusion)


if __name__ == "__main__":
    page()