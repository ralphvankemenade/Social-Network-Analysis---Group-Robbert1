"""Streamlit page: Community Clustering."""

import streamlit as st
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib.patches as mpatches

from dss.ui.state import init_state, get_state, set_state
from dss.analytics.communities import compute_communities
from dss.analytics.robustness import perturbation_test
from dss.ui.components import display_network, display_histogram, display_boxplot
from dss.analytics.roles import compute_roles


def page() -> None:
    st.set_page_config(page_title="Community Clustering", layout="wide")
    st.title("Community Clustering", help="""
    On this page the network can be divided up into multiple clusters.
    Each of these clusters have more communication within the cluster than outside of the cluster.
    It can be used to detect different factions of the network.
    """)
  
    init_state()
    G = get_state("graph")
    if G is None:
        st.info("No graph loaded.  Please upload a `.mtx` file on the Upload page.")
        return

    with st.expander("Quick User Guide", expanded=False):
        st.markdown("""
        ### Clustering Guide
        """)
            
        col_left, col_right = st.columns([3, 2])
    
        with col_left:
            st.markdown("""
            ### Community Summary
            
            **Modularity Q Score**  
            A measure of how well a network is partitioned into communities. 
            Value close to 1: indicates strong community structure.
            Value close to 0: indicates weak community structure.
            
            **Within Ratio**  
            A measure of how internally connected the communities are, as oposed to connections outside of the community.
            High within ratio: Community mostly communicates within the community.
            Low within ratio: Community interacts heavily with other communities.
            
            ### Clustering Methods
            
            **Girvan-Newman**  
            Detects communities by repeatedly removing edges with high betweenness centrality, which act as bridges between groups. 
            As these bridging edges are removed, the network splits into increasingly well-defined communities. 
            The amount of communities to be distinct can be adjusted.
            
            **Spectral**  
            Identifies communities by using the eigenvectors of the graph Laplacian to partition the network into weakly connected groups. 
            It is based on minimizing a graph-cut objective and is effective at revealing global structure in the network. 
            The amount of communities to be distinct can be adjusted.
            
            **Louvain**  
            Detects communities by iteratively grouping nodes to maximize the modularity Q score. 
            It is well suited for large networks and produces a hierarchical community structure.
            Because the Louvain method maximizes modularity it can not be used to set a number for the amount of communities you want to distinct. 
            Therefore it may make a distinction between more than two communities.
            """)
        with col_right:
            st.markdown("""
            
            ### Robustness Analysis
            
            Robustness analysis evaluates how stable the results of a network analysis are when the network is slightly altered or when different methods are applied. 
            A robust result indicates that the identified structure reflects meaningful patterns rather than noise or modeling choices.
            
            **Perturbation Test**  
            Assesses robustness by deliberately introducing small changes to the network, in this case removing some of the edges, and re-running the analysis. 
            If the results remain largely unchanged, the detected structure is considered robust.
            
            **Adjusted Rand Index (ARI)**  
            Measures the similarity between two clusterings while correcting for simularities that could occur by chance. 
            In this context, it is used to quantify how consistently communities are identified under network perturbations. 
            ARI close to 1 = Robust method of community clustering.
            ARI close to 0 = Community completely changes when perturbation test is applied
            """)
    
    # Sidebar: choose method and parameters
    st.sidebar.header("Community detection parameters")
    # method = st.sidebar.selectbox("Method", ["louvain", "girvan_newman", "spectral"], index=0)
    comm_method_labels = {
                "Spectral": "spectral",
                "Girvan Newman": "girvan_newman",
                "Louvain": "louvain"
                } 
    comm_label = st.sidebar.selectbox("Community method", list(comm_method_labels.keys()), 
                                      index=0, help = "Select method of computing community clusters.")
    method = comm_method_labels[comm_label]
    if method in {"girvan_newman", "spectral"}:
        k = st.sidebar.slider("Number of clusters (k)", 2, max(2, int(G.number_of_nodes() / 2)), 2, help = "Select the number of communities you want to distinct.")
    else:
        k = None
    # Compute communities

    # HIER MOET IETS AANGEPAST WORDEN DAT DE CLUSTERING JUIST WORDT GEUPDATE WANNEER DE SLIDER WORDT GEBRUIKT
    cache_key = (method, k)

    if get_state("community_results").get(cache_key) is None:
        comm_result = compute_communities(G, method=method, k=k)
        get_state("community_results")[cache_key] = comm_result
    comm_result = get_state("community_results")[cache_key]

    
 #   if get_state("community_results").get(method) is None:
 #       comm_result = compute_communities(G, method=method, k=k)
#        get_state("community_results")[method] = comm_result
 #   comm_result = get_state("community_results")[method]
    
    # Robustness analysis
    runs = st.sidebar.slider("Number of perturbation runs", 10, 100, 50, help = "Select amount of perturbation runs. more runs = more certainty.")
    p = st.sidebar.slider("Fraction of edges to remove", 0.01, 0.30, 0.05, 0.01, help = "Select fraction of edged to be removes, more removal = more drastic changes to network.")
    
    robustness_result = get_state("robustness_result")
    # Auto-run once on page load
    if robustness_result is None:
        robustness_result = perturbation_test(
            G,
            method=method,
            p=p,
            runs=runs,
            k=(k or 2),
        )
        set_state("robustness_result", robustness_result)
    # Optional manual rerun
    if st.sidebar.button("Run robustness test", help = "(re)run robustness test and update changes."):
        robustness_result = perturbation_test(
            G,
            method=method,
            p=p,
            runs=runs,
            k=(k or 2),
        )
        set_state("robustness_result", robustness_result)
    
    
   # if st.sidebar.button("Run robustness test", help = "(re)run robustness test and update changes."):
   #     robustness_result = perturbation_test(G, method=method, p=p, runs=runs, k=(k or 2))
   #     set_state("robustness_result", robustness_result)
   # robustness_result = get_state("robustness_result")

    # Allow user to select nodes for inspection
    # st.sidebar.subheader("Select nodes to inspect", )
    # selected_nodes = st.sidebar.multiselect(
    #     "Nodes", options=list(G.nodes()), default=[], help = "Select nodes based on number in the graph."
    # )
    selected_nodes = st.sidebar.multiselect(
        "Select nodes to inspect",
        options=list(G.nodes()),
        default=[],
        help="""
Select one or more nodes to inspect in detail.

Selected nodes will:
- Always be highlighted in the network view
- Appear in a detailed table at the bottom of this page
"""
    )
    # highlight_nodes = selected_nodes
    highlight_nodes_selected = list(selected_nodes)
    
    
    # Display summary
    col_stats, col_plot = st.columns(2)
    with col_stats:
        st.subheader("Community summary", 
                     help= """
Size = Amount of nodes in cluster

Within Ratio = A measure of how internally connected the communities are, as oposed to connections outside of the community. 

High within ratio: Community mostly communicates within the community. 

Low within ratio: Community interacts heavily with other communities.
"""
                    )
        st.write(f"Modularity Q: {comm_result.modularity:.3f}")
        st.dataframe(comm_result.summary)
        
    with col_plot:
        # Network plot coloured by communities with node selection
        community_colors = {node: comm_result.labels[node] for node in G.nodes()}
        st.subheader("Network coloured by communities", help= """

Visualisation of clustered network with selected method and parameters.

Different colors represent different communities.
"""
                    )   
        
        display_network(
            G,
            node_color=community_colors,
            highlight_selected=highlight_nodes_selected,
            title=f"Communities ({method})",
            show_labels=True,
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
        
    st.subheader("Robustness analysis", 
                 help= """

Robustness analysis evaluates how stable the results of a network analysis are when the network is slightly altered or when different methods are applied. 

A robust result indicates that the identified structure reflects meaningful patterns rather than noise or modeling choices.
"""
                )
    st.write("Click on Run Robustness Test in the side bar to (re)run robustness test")
    if robustness_result is not None:
        st.write(f"Average ARI across runs: {sum(robustness_result.ari_scores) / len(robustness_result.ari_scores):.3f}")
        st.write(f"Average modularity drop: {sum(robustness_result.modularity_drops) / len(robustness_result.modularity_drops):.3f}")
        col_hist, col_box = st.columns(2)

        with col_hist:
            display_histogram(
                robustness_result.ari_scores,
                title="ARI distribution",
                xlabel="ARI",
            )

        with col_box:
            display_boxplot(
                robustness_result.modularity_drops,
                title="Modularity drop distribution",
                ylabel="ΔQ",
            )
        
    # Compare to roles
#    st.subheader("Comparison with role clustering")
    # Compute role result if not already
#    if get_state("role_result") is None:
 #       from dss.analytics.centrality import compute_centralities
#        centrality_table = compute_centralities(G)
 #       role_result = compute_roles(G, centrality_table=centrality_table)
#        set_state("role_result", role_result)
#    role_result = get_state("role_result")
#    role_labels_list = [role_result.labels[node] for node in G.nodes()]
#    comm_labels_list = [comm_result.labels[node] for node in G.nodes()]
#    ari = adjusted_rand_score(role_labels_list, comm_labels_list)
#    nmi = normalized_mutual_info_score(role_labels_list, comm_labels_list)
#    st.write(f"Adjusted Rand Index between roles and communities: {ari:.3f}")
#    st.write(f"Normalized Mutual Information: {nmi:.3f}")
    
#    import pandas as pd  # imported here to avoid top-level import issues
#    confusion = pd.crosstab(pd.Series(role_labels_list, name="role"), pd.Series(comm_labels_list, name="community"))
#    st.dataframe(confusion)


if __name__ == "__main__":
    page()
