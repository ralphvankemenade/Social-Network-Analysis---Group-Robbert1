# """Streamlit page: Arrest optimisation model."""

# import streamlit as st
# import pandas as pd

# from dss.ui.state import init_state, get_state, set_state
# from dss.analytics.communities import compute_communities
# from dss.analytics.centrality import compute_centralities, combine_centralities
# from dss.analytics.arrest_optimization import arrest_assignment
# from dss.ui.components import display_network


# def page() -> None:
#     st.set_page_config(page_title="Arrest Optimisation", layout="wide")
#     st.title("Arrest Optimisation")
#     init_state()
#     G = get_state("graph")
#     if G is None:
#         st.info("No graph loaded.  Please upload a `.mtx` file on the Upload page.")
#         return
#     # Sidebar parameters
#     st.sidebar.header("Arrest optimisation parameters")
#     comm_method = st.sidebar.selectbox("Community method for regret weights", ["louvain", "girvan_newman", "spectral"], index=0)
#     alpha = st.sidebar.slider("Regret strength (alpha)", 0.0, 5.0, 1.0, 0.1)
#     beta = st.sidebar.slider("Penalty strength (beta)", 0.0, 5.0, 1.0, 0.1)
#     centrality_metric = st.sidebar.selectbox("Centrality for regret", ["None", "degree", "combined"], index=2)
#     if st.sidebar.button("Compute arrest assignment") or get_state("arrest_result") is None:
#         # Compute community labels
#         if get_state("community_results").get(comm_method) is None:
#             comm_result = compute_communities(G, method=comm_method, k=2)
#             get_state("community_results")[comm_method] = comm_result
#         comm_result = get_state("community_results")[comm_method]
#         communities = comm_result.labels
#         # Centrality scores
#         if centrality_metric == "None":
#             centrality_scores = None
#         else:
#             if get_state("centrality_result") is None:
#                 from dss.analytics.centrality import compute_centrality_result
#                 result = compute_centrality_result(G)
#                 set_state("centrality_result", result)
#             centrality_table = get_state("centrality_result").table
#             if centrality_metric == "combined":
#                 combined = combine_centralities(centrality_table)
#                 centrality_scores = combined
#             else:
#                 centrality_scores = centrality_table[centrality_metric]
#         # Compute assignment
#         arrest_result = arrest_assignment(G, communities, centrality_scores, alpha=alpha, beta=beta)
#         set_state("arrest_result", arrest_result)
#     arrest_result = get_state("arrest_result")
#     if arrest_result is not None:
#         st.subheader("Optimisation results")
#         st.write(f"Objective value: {arrest_result.objective:.3f}")
#         st.write(f"Cross‑department edges: {arrest_result.cut_edges}")
#         st.write(f"Estimated effective arrests: {arrest_result.effective_arrests:.3f}")
#         # Display network coloured by department with labels shown.  Nodes
#         # assigned to different departments are coloured differently.  Labels
#         # allow you to identify specific individuals.
#         dept_colors = {node: arrest_result.assignment[node] for node in G.nodes()}
#         display_network(
#             G,
#             node_color=dept_colors,
#             title="Department assignment (0/1)",
#             show_labels=True,
#         )
#         # Show list of risky edges
#         if arrest_result.risk_edges:
#             df_risk = pd.DataFrame(arrest_result.risk_edges, columns=["u", "v"])
#             st.subheader("Edges across departments (risk)")
#             st.dataframe(df_risk)
#     else:
#         st.info("No optimisation result available yet.  Adjust parameters and press the button to compute.")


# if __name__ == "__main__":
#     page()
"""Streamlit page: Arrest optimisation model."""

import streamlit as st
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from dss.ui.state import init_state, get_state, set_state
from dss.analytics.communities import compute_communities
from dss.analytics.arrest_optimization import arrest_assignment
from dss.ui.components import display_network
from dss.analytics.centrality import combine_centralities, borda_count, compute_centrality_result


def page() -> None:
    st.set_page_config(page_title="Arrest Optimisation", layout="wide")
    st.title("Arrest Optimisation")
    init_state()
    G = get_state("graph")
    if G is None:
        st.info("No graph loaded.  Please upload a `.mtx` file on the Upload page.")
        return

    with st.expander("Quick User Guide", expanded=False):
        st.markdown(
            """
         
            """,
        )
    
        **Estimated number of effective arrests**   
        The effective arrests are  the number of arrests possible that realistically can be carried out, considering the cross department edges.  
        
        **Visualisation**  
        The network shows the result of the arrest optimisation. Nodes are colourd by deparment assignment, and the risky edges are highligted in red.
        This helps indentify critcal members and risky edges.  
        """
        )
            
            st.markdown("### Edges across departments")
            st.markdown(
        """
        These edges are considered risky because they connect members from different departments.

        **Table**  
        This table list the risky edges connecting members from department 1 to department 2. 

                    """)
        with col_right:
            st.markdown("### Adjust paramets in sidebar")
            with st.expander("Centrality Methods Details", expanded=False):
                st.markdown("""
               
                """)

            with st.expander("Clustering Methods Details", expanded=False):
                st.markdown("""
               """)

    if get_state("centrality_result") is None:
        centrality_result = compute_centrality_result(G)
        set_state("centrality_result", centrality_result)
    else:
        centrality_result = get_state("centrality_result")
    df = centrality_result.table

    # Sidebar parameters
    st.sidebar.header("Optimisation parameters")
    comm_method_labels = {"Spectral": "spectral",
                "Girvan Newman": "girvan_newman",
                "Louvain": "louvain"
                }
    comm_label = st.sidebar.selectbox("Community method", list(comm_method_labels.keys()), index=0, help ="Select the method that detects the communities in the network.")
    comm_method = comm_method_labels[comm_label]
    alpha = st.sidebar.slider("Importance Communities (alpha)", 0.0, 5.0, 1.0, 0.1,help="How important it is to keep members of the same community together. Higher values make the model more likely to keep them together.")
    beta = st.sidebar.slider("Importance centrality (beta)", 0.0, 5.0, 1.0, 0.1, help = "How important it is to arrest members with a high centrality score. Higher values make the model more likely to arrest members with a high centrality score.")
    
    combined = None
    centrality_metric = None
    weight_inputs = {}
    
    cent_method = st.sidebar.radio("Centrality type", ["Single method", "Combined method"], index=0, help = "Select how centrality scores should be computed: individual metric or a combination of metrics.")
    # Node selection for detailed view
    if cent_method == "Single method":
        centrality_metric_labels_single = {
                    "Degree": "degree",
                    "Eigenvector": "eigenvector",
                    "Katz" : "katz",
                    "Betweenness": "betweenness", 
                    "Closeness": "closeness", 
                    "Pagerank" : "pagerank"
                    }
        centrality_metric_single = st.sidebar.selectbox("Single centrality method", list(centrality_metric_labels_single.keys()), index=2, help = "Select the method that determines the centrality of the members of the network.")
        centrality_metric = centrality_metric_labels_single[centrality_metric_single]
    
    else: #Combined methods
        centrality_metric_labels_combined = {
            "Weighted sum": "Weighted sum",
            "Borda count": "Borda count"
                    }
        centrality_metric = st.sidebar.selectbox("Combined centrality method", list(centrality_metric_labels_combined.keys()), index=0, help = "Select the method that determines the centrality of the members of the network.")
        centrality_metric = centrality_metric_labels_combined[centrality_metric]

        if centrality_metric == "Weighted sum":
            st.sidebar.header("Weighting scheme", help = "How imporant each centrality method is. Higer = more imporant.")
            for col in df.columns:
                weight_inputs[col] = st.sidebar.slider(f"Weight for {col}", 0.0, 1.0, 1.0, 0.1)
            combined = combine_centralities(df, weights=weight_inputs)
        
        else: #Borda count
            # Initialize all toggles to True the first time we enter Borda mode
            if not st.session_state.get("borda_toggles_initialized", False):
                for col in df.columns:
                    st.session_state[f"borda_use_{col}"] = True
                st.session_state.borda_toggles_initialized = True
            st.sidebar.markdown(
                "Toggle which metrics should be included in the Borda count aggregation. "
                   )
                   
            for col in df.columns:
                key = f"borda_use_{col}"
                weight_inputs[col] = st.sidebar.toggle(label=str(col), key=key)

            combined = borda_count(df, weight_inputs)
    st.sidebar.markdown("<div style='margin:0;padding:0'></div>", unsafe_allow_html=True)
    selected_nodes = st.sidebar.multiselect(
        "Select nodes to inspect", options=list(G.nodes()), default=[]
        )
    
    # if st.sidebar.button("Compute arrest assignment") or get_state("arrest_result") is None:
        # Compute community labels
    if get_state("community_results").get(comm_method) is None:
        comm_result = compute_communities(G, method=comm_method, k=2)
        get_state("community_results")[comm_method] = comm_result
    comm_result = get_state("community_results")[comm_method]
    communities = comm_result.labels
    
    # Centrality scores
    if cent_method == "Single method":
        centrality_scores = df[centrality_metric]
    else:
        centrality_scores = combined
    # Compute assignment
    arrest_result = arrest_assignment(G, communities, centrality_scores, alpha=alpha, beta=beta)
    set_state("arrest_result", arrest_result)
    arrest_result = get_state("arrest_result")
    if arrest_result is not None:
        st.write("")
        st.subheader("Optimisation results")
        st.write(f"**Objective value:** {arrest_result.objective:.3f}")
        st.write(f"**Estimated number of effective arrests:** {arrest_result.effective_arrests:.1f}")
        
        # Display network coloured by department with labels shown.  Nodes
        # assigned to different departments are coloured differently.  Labels
        # allow you to identify specific individuals.
        dept_colors = {node: arrest_result.assignment[node] for node in G.nodes()}
        highlight_nodes_selected = list(selected_nodes)
        display_network(
            G,
            node_color=dept_colors,
            title="Department assignment",
            #add risky edges to plot 
            removed_edges=arrest_result.risk_edges,
            show_labels=True,
            # highlight=selected_nodes
            highlight_selected=highlight_nodes_selected,
        )

        team_colors = { 0: "#440154", 1: "#FDE725" }
        team_patches = [mpatches.Patch(color=color, label=f"department{team + 1}") for team, color in team_colors.items()]
        risky_line = mlines.Line2D([], [], color='red', linestyle='dashed', label='Risky edges')
        legend_items = team_patches + [risky_line]

        # legend
        plt.legend(handles=legend_items, loc="upper right")
        st.pyplot(plt.gcf())

        # Show list of risky edges
        if arrest_result.risk_edges:
            df_risk = pd.DataFrame(arrest_result.risk_edges, columns=["u", "v"])
            st.subheader("Edges across departments")
            st.write(f"Number of Cross‑department edges: {arrest_result.cut_edges}")
            df_risk["Member department 1"] = df_risk.apply(lambda row: row["u"] if arrest_result.assignment[row["u"]] == 0 else row["v"], axis=1)
            df_risk["Member department 2"] = df_risk.apply(lambda row: row["u"] if arrest_result.assignment[row["u"]] == 1 else row["v"], axis=1)
            df_risk = df_risk[["Member department 1", "Member department 2"]]
            df_risk = df_risk.sort_values(["Member department 1","Member department 2"])
            st.markdown("###### Risky edges connecting members from  department 1 to department 2. ")
            st.dataframe(df_risk)

    else:
        st.info("No optimisation result available yet.  Adjust parameters and press the button to compute.")
    if selected_nodes:
        # Df of selected nodes
        risky_edges = set([u for u, v in arrest_result.risk_edges] + [v for u, v in arrest_result.risk_edges])
    
        selected_df = pd.DataFrame({
            "Node": selected_nodes,
            "Assigned department": [arrest_result.assignment[node] + 1 for node in selected_nodes],
            "Risky edges count": [sum(node in edge for edge in arrest_result.risk_edges) for node in selected_nodes]
        })
        st.subheader("Selected node details")
        st.dataframe(selected_df)

if __name__ == "__main__":
    page()
