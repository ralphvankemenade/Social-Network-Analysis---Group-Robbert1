"""Streamlit page: Role identification via similarity clustering."""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from dss.ui.state import init_state, get_state, set_state
from dss.analytics.roles import compute_roles
from dss.analytics.centrality import compute_centralities
from dss.ui.components import display_network, display_heatmap
from dss.analytics.communities import compute_communities


def page() -> None:
    st.set_page_config(page_title="Role Analysis", layout="wide")
    st.title("Role Identification via Similarity Clustering")
    init_state()
    G = get_state("graph")
    if G is None:
        st.info("No graph loaded.  Please upload a `.mtx` file on the Upload page.")
        return
    # Sidebar parameters
    st.sidebar.header("Role similarity parameters")
    signature = st.sidebar.selectbox("Structural signature", ["k-hop", "random-walk"], index=0)
    if signature == "k-hop":
        k = st.sidebar.slider("Max hop (k)", 1, 5, 3)
        t = 3
    else:
        t = st.sidebar.slider("Random-walk steps (t)", 1, 5, 3)
        k = 3
    similarity_metric = st.sidebar.selectbox("Similarity metric", ["cosine", "correlation"], index=0)
    clustering_method = st.sidebar.selectbox("Clustering method", ["spectral", "hierarchical"], index=0)
    auto_clusters = st.sidebar.checkbox("Auto-detect number of clusters", value=True)
    if auto_clusters:
        n_clusters = None
    else:
        n_clusters = st.sidebar.slider("Number of clusters", 2, max(2, int(np.ceil(np.sqrt(G.number_of_nodes())))), 4)
    compute_button = st.sidebar.button("Compute roles")
    if compute_button or get_state("role_result") is None:
        # Compute centralities for summary statistics
        centralities = compute_centralities(G)
        role_result = compute_roles(
            G,
            signature=signature,
            k=k,
            t=t,
            similarity_metric=similarity_metric,
            clustering_method=clustering_method,
            n_clusters=n_clusters,
            centrality_table=centralities,
        )
        set_state("role_result", role_result)
    else:
        role_result = get_state("role_result")
    # Display similarity heatmap
    st.subheader("Role similarity heatmap")
    display_heatmap(role_result.similarity_matrix, list(G.nodes()), caption="Role similarity")
    # Display role summary
    st.subheader("Role cluster summary")
    st.dataframe(role_result.summary)
    # Colour map for roles
    role_colors = {node: role_result.labels[node] for node in G.nodes()}
    # Plot network coloured by roles with labels and interactive highlights
    st.subheader("Network coloured by roles")
    # Node selection for highlight and inspection
    st.sidebar.subheader("Select nodes to inspect")
    selected_nodes = st.sidebar.multiselect(
        "Nodes", options=list(G.nodes()), default=[]
    )
    # Highlight nodes that are selected
    highlight_nodes = selected_nodes
    # Build hover text: include role cluster and centrality values
    centralities_all = compute_centralities(G)
    hover_dict = {}
    for node in G.nodes():
        parts = [f"Role: {role_result.labels[node]}"]
        for col in centralities_all.columns:
            parts.append(f"{col}: {centralities_all.at[node, col]:.4f}")
        hover_dict[node] = "<br>".join(parts)
    display_network(
        G,
        node_color=role_colors,
        highlight=highlight_nodes,
        hover=hover_dict,
        title="Roles",
        label_dict=None,
    )
    # Show details for selected nodes
    if selected_nodes:
        st.subheader("Selected node details")
        # Build a DataFrame with role label and basic centrality measures
        data = centralities_all.loc[selected_nodes].copy()
        data["role_cluster"] = [role_result.labels[n] for n in selected_nodes]
        st.dataframe(data)
    # Compare roles to communities if available
    st.subheader("Comparison with community clustering")
    comm_method = st.selectbox("Community method for comparison", ["louvain", "girvan_newman", "spectral"], index=0)
    # Compute community result (cached per method)
    if get_state("community_results").get(comm_method) is None:
        comm_result = compute_communities(G, method=comm_method, k=2)
        get_state("community_results")[comm_method] = comm_result
    comm_result = get_state("community_results")[comm_method]
    # Compute ARI and NMI between role labels and community labels
    role_labels_list = [role_result.labels[node] for node in G.nodes()]
    comm_labels_list = [comm_result.labels[node] for node in G.nodes()]
    ari = adjusted_rand_score(role_labels_list, comm_labels_list)
    nmi = normalized_mutual_info_score(role_labels_list, comm_labels_list)
    st.write(f"Adjusted Rand Index between roles and communities: {ari:.3f}")
    st.write(f"Normalized Mutual Information: {nmi:.3f}")
    # Confusion matrix
    df_conf = pd.DataFrame(
        {
            "role": role_labels_list,
            "community": comm_labels_list,
        }
    )
    confusion = pd.crosstab(df_conf["role"], df_conf["community"])
    st.dataframe(confusion)


if __name__ == "__main__":
    page()