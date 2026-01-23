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

    with st.expander("Quick User Guide", expanded=False):
        st.markdown(
            """
            **Centrality metrics (what do they mean?)**
            """,
        )

        col_left, col_right = st.columns([3, 2], gap="large")

        with col_left:
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

        with col_right:
            st.markdown("#### How the final score is built")
            st.markdown(
                """
                    **Weighting scheme**  
                    Use the sliders to decide which metrics matter most.  
                    Higher weight means more influence on the final score.

                    ---
                    
                    **Aggregation method**
                    
                    **Weighted sum**  
                    Combines all metrics using your chosen weights.
                    
                    **Borda count**  
                    Ranks nodes per metric and combines the rankings.  
                    Useful if you care about rank agreement rather than raw score magnitude.

                    ---
                    """
            )

    # Compute or retrieve centrality result
    if get_state("centrality_result") is None:
        result = compute_centrality_result(G)
        set_state("centrality_result", result)
    else:
        result = get_state("centrality_result")

    df = result.table
    combined_scores = result.combined_scores
    ranks = result.ranks

    col_left, col_right = st.columns([2, 2], gap="small")
    with col_left:
        st.subheader(
            "Centrality measures",
            help="""
This table shows the centrality scores per node.

Each column is one centrality metric.
The 'combined' column is the final aggregated score based on the settings in the sidebar.

Tip:
Sort by 'combined' to see which nodes are most central according to your chosen method.
    """,
        )
        weight_inputs = {}

        # Choose aggregation method
        agg_method = st.sidebar.radio(
            "Aggregation method",
            ["Weighted sum", "Borda count"],
            index=0,
            key="centrality_agg_method",
            help="""
Choose how the individual centrality metrics are combined into one final score.

Weighted sum:
- Uses your slider weights directly.
- Higher weight means that metric contributes more to the final score.

Borda count:
- Turns each metric into a ranking (best to worst).
- Combines those rankings into a single overall ranking.
    """,
        )

        # Reset Borda toggles every time we switch into Borda mode
        prev_method = st.session_state.get("centrality_prev_agg_method", None)
        if agg_method == "Borda count" and prev_method != "Borda count":
            for col in df.columns:
                st.session_state[f"borda_use_{col}"] = True

        st.session_state["centrality_prev_agg_method"] = agg_method

        if agg_method == "Weighted sum":
            # Sidebar for weighting scheme
            st.sidebar.header("Weighting scheme")
            for col in df.columns:
                weight_inputs[col] = st.sidebar.slider(
                    f"Weight for {col}",
                    0.0,
                    1.0,
                    1.0,
                    0.1,
                    help=f"""
Controls how much *{col}* influences the final combined score.

0.0  means this metric is ignored.
1.0  means this metric is fully included (relative to the other weights).

Tip:
If you want the final score to reflect mostly one metric, set that one high
and set the others close to 0.
    """,
                )
            combined = combine_centralities(df, weights=weight_inputs)
        else:
            st.sidebar.header("Measure scheme")

            for col in df.columns:
                key = f"borda_use_{col}"
                weight_inputs[col] = st.sidebar.toggle(
                    label=str(col),
                    key=key,
                    help=f"""
Include or exclude *{col}* from the Borda aggregation.

On:
- This metric is used to create a ranking and contributes to the final result.

Off:
- This metric is ignored completely in the Borda calculation.
    """,
                )

            combined = borda_count(df, weight_inputs)

        combined.index.name = "Node"
        # Display centrality table
        st.dataframe(
            df.assign(combined=combined).sort_values("combined", ascending=False),
        )

        # Download as CSV
        csv_data = df.assign(combined=combined).to_csv().encode("utf-8")
        st.download_button(
            "Download centrality data as CSV",
            csv_data,
            file_name="centrality.csv",
            mime="text/csv",
            help="""
Download the current centrality table as a CSV file.

The exported file includes:
- All centrality metric columns
- The current 'combined' score based on your sidebar settings

Use this if you want to analyze the results outside the dashboard.
""",
        )

        # Highlight controls and node selection
        st.sidebar.header("Highlight and select nodes")
        max_n = min(20, len(df))
        top_n = st.sidebar.slider(
            "Top N",
            1,
            max_n,
            min(5, max_n),
            help="""
Choose how many nodes to highlight as the most or least central.

Example:
If Top N = 5 and "Highlight top N" is enabled, the 5 highest combined-score nodes
will be visually highlighted in the network.
    """,
        )
        highlight_top = st.sidebar.checkbox(
            "Highlight top N",
            value=True,
            help="""
Highlights the Top N nodes with the highest combined centrality score in the network view.

Use this to quickly identify the most central or influential nodes under your current settings.
    """,
        )
        highlight_bottom = st.sidebar.checkbox(
            "Highlight bottom N",
            value=False,
            help="""
Highlights the Top N nodes with the lowest combined centrality score in the network view.

This can help you spot peripheral or weakly connected nodes.
    """,
        )

        # Node selection for detailed view
        selected_nodes = st.sidebar.multiselect(
            "Select nodes to inspect",
            options=list(G.nodes()),
            default=[],
            help="""
Select one or more nodes to inspect in detail.

Selected nodes will:
- Always be highlighted in the network view
- Appear in a detailed table at the bottom of this page
    """,
        )

        # Determine highlight nodes: top/bottom plus selected
        highlight_nodes = []
        if highlight_top:
            highlight_nodes += combined.nlargest(top_n).index.tolist()
        if highlight_bottom:
            highlight_nodes += combined.nsmallest(top_n).index.tolist()

        # Always include explicitly selected nodes in highlight list
        # highlight_nodes += [n for n in selected_nodes if n not in highlight_nodes]
        highlight_nodes_selected = list(selected_nodes)

        with col_right:
            st.subheader(
                "Network with centrality-aggregated nodes",
                help="""
This network visualization shows the graph with node size scaled by the aggregated centrality score.

Each node represents an entity in the network. Larger nodes have a higher combined centrality score, meaning they are more important or influential according to the selected aggregation method and weights. Smaller nodes are less central under the same settings.

Node colors reflect relative centrality levels, helping you visually distinguish highly central nodes from more peripheral ones. Edges represent connections between nodes.

Nodes outlined in red are highlighted based on your sidebar selections, such as the Top N most central nodes, the Bottom N least central nodes, or nodes you selected manually. Highlighted nodes always remain visible, even if multiple highlighting rules apply.

Use this view to quickly identify key hubs, bridges, and structurally important nodes, and to understand how changes in weighting or aggregation affect the perceived importance of nodes in the network.
        """,
            )

            # Map node sizes and colours
            size_map = combined.to_dict()
            color_map = combined.to_dict()

            display_network(
                G,
                node_size=size_map,
                node_color=color_map,
                highlight_top=highlight_nodes,
                highlight_selected=highlight_nodes_selected,
                title="Centrality-scaled network",
                show_labels=True,
            )

    # Show information for selected nodes
    if selected_nodes:
        st.subheader(
            "Selected node details",
            help="""
This section shows the exact metric values for the nodes you selected in the sidebar.

Use it to compare nodes side-by-side and understand *why* a node scores high or low
under different metrics and aggregation settings.
""",
        )
        info_df = df.loc[selected_nodes].copy()
        info_df["combined"] = combined.loc[selected_nodes]
        info_df.index.name = "Node"
        st.dataframe(
            info_df,
        )


if __name__ == "__main__":
    page()





