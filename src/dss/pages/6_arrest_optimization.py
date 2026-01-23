"""Streamlit page: Arrest optimisation model."""

import streamlit as st
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.lines import Line2D
import matplotlib as mpl

from dss.ui.state import init_state, get_state, set_state
from dss.analytics.communities import compute_communities
from dss.analytics.arrest_optimization import arrest_assignment,compute_arrest_order,simulate_sequential_arrests
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
        col_left, col_right = st.columns([3, 2], gap="large")

        with col_left:
            st.markdown("### Arrest Optimisation")
            st.markdown(
        """
        This page computes an optimal assignment of network of members to departments. The goal is to improve the effectiveness of arrests by:
        -	Arresting highly central members
        -	Keeping members of the same community together
        -	Minimizing risky connections between departments
         """
    )
            st.markdown("### Optimisation results")
            st.markdown(
        """
        **Objective value**  
        The objective value indicates how good the optimisation run is. Lower values mean a better result. However they are relative and the values cannot be used to be compare different networks.
        The objective value is determined by:
        -	Alpha: importance of keeping member of the same Communities  together
        -	Beta: importance of arresting highly central members  
        Higher weight means more influence on the final optimisation score. 
        
        **Estimated number of effective arrests**   
        The effective arrests are the number of arrests possible that realistically can be carried out, considering the cross department edges, because information may leak between departments. 
        
        **Visualization**  
        The network shows the result of the arrest optimisation. Nodes are colored by department assignment, and the risky edges are highlighted in red. If there are optional nodes to inspect, they are highlighted in pink and have a table on the bottom with their assigned department and number of risky edges. 
        This visualization helps identify critical members and risky edges.  
          """
    )
            st.markdown("### Edges across departments")
            st.markdown("Edges are considered risky because they connect members from different departments and may reduce the effectiveness of arrests. The table list the all risky edges connecting members from departments.")   
      
            st.markdown("### Recommended arrest order")
            st.markdown(
        """
          The total effective arrests are the minimum number of arrests possible when executing the arrests in the order stated in the table. It is the minimum because members who are possibly tipped may not be arrested. The visualization shows the arrested members highlighted in green and the potential tipped members in red.
          It is possible to download the table as a CSV-file.
          """
    )       
        with col_right:
            st.markdown("### Optimisation parameters")
            st.markdown(
        """
            Choose the preferred method to detect communities in the network and adjust the sliders.
          """
            )

            with st.expander("Details Community Methods ", expanded=False):
                st.markdown(
                """
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

                """
            )

            st.markdown(
                """
            Centrality type:
            -	Single method: choose one method to determine the centrality of the members 
            -	Combined method: use multiple methods to determine the centrality of the members. 
                -	Weighted sum, adjust the sliders to select the importance of each method 
                -	Borda count toggle which methods should be used in the ranking

            """
            )
        
            with st.expander("Details Centrality Methods ", expanded=False):
                st.markdown(
                    """
                **Degree**  
                How many direct connections does a node have?  
                High score = very connected or popular.  
                When useful: spotting hubs that connect to many neighbors.
                
                **Eigenvector**  
                Are you connected to other important nodes?  
                High score = influence through influential connections.  
                When useful: finding nodes that sit inside powerful neighborhoods.
                
                **Katz**  
                How far does your influence reach, directly and indirectly?  
                High score = strong reach through the network.  
                When useful: capturing indirect influence beyond direct neighbors.
                
                **Betweenness**  
                How often do others need you to connect?  
                High score = you act as a bridge between groups.  
                When useful: identifying brokers and critical connectors.
                
                **Closeness**  
                How quickly can you reach everyone else?  
                High score = centrally located in terms of distance.  
                When useful: finding nodes with fast access across the network.
                
                **PageRank**  
                How much important attention flows to you?  
                High score = prestige or authority in the network.  
                When useful: detecting nodes that receive endorsement from other important nodes.
                    """
                )

    if get_state("centrality_result") is None:
        centrality_result = compute_centrality_result(G)
        set_state("centrality_result", centrality_result)
    else:
        centrality_result = get_state("centrality_result")
    df = centrality_result.table

    # Sidebar parameters
    st.sidebar.header("Optimisation parameters")
    if G.is_directed():
        comm_method_labels = {
                "Spectral": "spectral",
                "Girvan Newman": "girvan_newman",
                } 
    else:
        comm_method_labels = {
                    "Spectral": "spectral",
                    "Girvan Newman": "girvan_newman",
                    "Louvain": "louvain"
                    } 
    # comm_method_labels = {"Spectral": "spectral",
    #             "Girvan Newman": "girvan_newman",
    #             "Louvain": "louvain"
    #             }
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
    
    else:  #Combined methods
        centrality_metric_labels_combined = {
            "Weighted sum": "Weighted sum",
            "Borda count": "Borda count"
                    }
        centrality_metric = st.sidebar.selectbox("Combined centrality method", list(centrality_metric_labels_combined.keys()), index=0, help = "Select the method that determines the centrality of the members of the network.")
        centrality_metric = centrality_metric_labels_combined[centrality_metric]

        if centrality_metric == "Weighted sum":
            st.sidebar.header("Weighting scheme", help = "How important each centrality method is. Higer = more important.")
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
        col_left, col_right = st.columns([3, 2], gap="large")
        
        with col_left:
            st.subheader("Optimisation results", help = "The objective value indicates how good the optimisation run is. Lower values mean a better result. The effective arrests are the number of arrests possible that realistically can be carried out, considering the cross department edges." )

            current_objective = arrest_result.objective
            last_objective = st.session_state.get("last_objective")

            if last_objective is not None:
                delta_objective = current_objective - last_objective
                st.metric(
                    label="Objective value", value=f"{current_objective:.3f}",
                    delta=f"{delta_objective:+.3f}",
                    delta_color="inverse",  # lager is beter -> groen bij negatieve delta
                )

            else:
                st.metric(label="Objective value", value=f"{current_objective:.3f}",delta=None)
            
            st.session_state["last_objective"] = current_objective
        
            st.write(f"**Estimated number of effective arrests:** {arrest_result.effective_arrests:.1f}")
            
            # Display network coloured by department with labels shown.  Nodes
            # assigned to different departments are coloured differently.  Labels
            # allow you to identify specific individuals.
            dept_colors = {node: arrest_result.assignment[node] for node in G.nodes()}
            highlight_nodes_selected = list(selected_nodes)

            team_colors = { 0: "#440154", 1: "#FDE725" }
            team_patches = [mpatches.Patch(color=color, label=f"Dept. {team + 1}") for team, color in team_colors.items()]
            risky_line = mlines.Line2D([], [], color='red', linestyle='dashed', label='Risky edge')
            legend_items = team_patches + [risky_line]
        
            display_network(
                G,
                node_color=dept_colors,
                title="Department assignment",
                #add risky edges to plot 
                removed_edges=arrest_result.risk_edges,
                show_labels=True,
                # highlight=selected_nodes
                highlight_selected=highlight_nodes_selected,
                legend_items = legend_items,
            )

        with col_right:
            # Show list of risky edges
            if arrest_result.risk_edges:
                df_risk = pd.DataFrame(arrest_result.risk_edges, columns=["u", "v"])
                st.subheader("Edges across departments", help = "Edges across departments are considered risky because they connect members from different departments and may reduce the effectiveness of arrests.")
                st.write(f"Number of Cross‑department edges: {arrest_result.cut_edges}")
                df_risk["Member dept. 1"] = df_risk.apply(lambda row: row["u"] if arrest_result.assignment[row["u"]] == 0 else row["v"], axis=1)
                df_risk["Member dept. 2"] = df_risk.apply(lambda row: row["u"] if arrest_result.assignment[row["u"]] == 1 else row["v"], axis=1)
                df_risk = df_risk[["Member dept. 1", "Member dept. 2"]]
                df_risk = df_risk.sort_values(["Member dept. 1","Member dept. 2"])
                df_risk = df_risk.set_index("Member dept. 1")
                st.dataframe(df_risk)

        arrest_order_df = compute_arrest_order(
                G,
                assignment=arrest_result.assignment,
                centrality=centrality_scores,
                risk_edges=arrest_result.risk_edges,
                gamma=1.0,
            )
        simulation_df = simulate_sequential_arrests(
                    G,
                    arrest_order_df,
                    arrest_result.risk_edges,
                )
        st.subheader("Recommended arrest order", help ="The order of the arrest has effect on the number of possible arrests. This is based on the centrality score and the number of risky edges.") 
        col_left, col_right = st.columns([2, 2], gap="large") 
        with col_right:
            st.metric("Total effective arrests", f"{(simulation_df['Status'] == 'Arrested').sum()}")
            display_df = arrest_order_df.copy()
            display_df["Tipped members"] = simulation_df["Tipped members"] 
            display_df["Status"] = simulation_df["Status"]
            cols_to_show = ["Arrest order","Node", "Dept.", "Tipped members", "Status"]
            display_df = display_df[cols_to_show]
            display_df = display_df.set_index("Arrest order")

            st.dataframe(display_df)

            # Download as CSV
            csv_data = display_df.to_csv(index=True).encode("utf-8")
            st.download_button(
                "Download recommended arrest order as CSV",
                csv_data,
                file_name="Arrest_order.csv",
                mime="text/csv",
                help="""
                    Download the current receommended arrest order table as a CSV file.
                    
                    Use this if you want to analyze the results outside the dashboard.
                    """,
            )

        with col_left:
            #plot 
            arrested_nodes = simulation_df.loc[simulation_df["Status"] == "Arrested", "Node"].tolist()
            tipped_nodes = simulation_df.loc[simulation_df["Status"] == "Tipped", "Node"].tolist()
            node_color_default = {node: 0 for node in G.nodes()}  

            display_network(
                G,
                node_color=node_color_default,  # alles standaard
                title= "Arrest order outcome",
                show_labels=True,
                highlight_top = tipped_nodes,
                highlight_arrested=arrested_nodes,
                legend_items=[
                    Line2D([0], [0], marker='o', color='w', label='Arrested', markerfacecolor='#0bd63e', markersize=10),
                    Line2D([0], [0], marker='o', color='w', label='Tipped', markerfacecolor="#FF0000", markersize=10),
                    ] 
            )
            
        if selected_nodes:
                # Df of selected nodes
                selected_df = simulation_df[simulation_df["Node"].isin(selected_nodes)].copy()
                selected_df["Assigned dept."] = selected_df["Node"].map(lambda node: arrest_result.assignment[node] + 1)
                selected_df["Risky edges count"] = selected_df["Node"].map(
                    lambda node: sum(node in edge for edge in arrest_result.risk_edges)
                )
                selected_df = selected_df[["Node", "Assigned dept.","Arrest order", "Risky edges count", "Tipped members", "Status"]]

                st.subheader("Selected nodes details")
                st.dataframe(selected_df)
    else:
        st.info("No optimisation result available yet.  Adjust parameters and press the button to compute.")
        

if __name__ == "__main__":
    page()
