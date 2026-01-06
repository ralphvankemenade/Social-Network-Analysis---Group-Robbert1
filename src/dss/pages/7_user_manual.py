"""Streamlit page: User Manual for the DSS."""

import streamlit as st


def page() -> None:
    st.set_page_config(page_title="User Manual", layout="wide")
    st.title("User Manual")
    st.markdown(
        """
        ## Introduction
        
        This Decision Support System (DSS) helps you analyse clandestine networks and make
        informed decisions about operational strategies.  The dashboard is organised
        into several pages, each focused on a specific analytic task.  This manual
        explains how to use the DSS and how to interpret its outputs.
        
        ## Upload & Overview
        
        1. Navigate to the **Upload & Overview** page.
        2. Use the file uploader to select a `.mtx` file containing the adjacency matrix of the network.
        3. After uploading, the DSS will validate the file, compute basic statistics
           (number of nodes, edges, density, connected components) and display a
           baseline network plot.  Any warnings about symmetry or self loops
           indicate potential data issues.
        
        ## Centrality Analysis
        
        Centrality measures quantify the importance of each node in the network.
        This page computes several centralities (degree, Katz, eigenvector,
        betweenness, closeness and PageRank) and allows you to:
        
        * **Weight measures:** Use the sliders in the sidebar to adjust the importance of each metric in the aggregated score.
        * **Aggregate metrics:** Choose between a weighted sum or a Borda count (rank aggregation) to combine measures.
        * **Highlight nodes:** Highlight the top and/or bottom `N` nodes based on the aggregated score.
        * **Select nodes:** Choose specific nodes from a list to inspect.  Selected nodes are highlighted on the network plot and their centrality values and aggregated score are shown in a table.
        * **Download data:** Export the centrality table as a CSV for offline analysis.
        
        Operational interpretation:
        
        * **Degree centrality:** Popularity of a node (number of connections).
        * **Katz centrality:** Accounts for all paths in the network, giving less weight to longer paths; useful for identifying influential spreaders.
        * **Eigenvector centrality:** Measures influence of a node in terms of its connections to other influential nodes.
        * **Betweenness centrality:** Captures brokerage power; nodes with high betweenness lie on many shortest paths.
        * **Closeness centrality:** Inverse of the average distance to all other nodes; smaller values indicate quick reachability.
        * **PageRank:** Probability of visiting a node in a random walk with teleportation; similar to eigenvector but more robust.
        
        ## Role Identification
        
        This page applies the role‑similarity approach of Cooper & Barahona (2010).
        Nodes with similar structural signatures (k‑hop neighbourhoods or
        random‑walk profiles) are grouped into roles.  You can choose:
        
        * **Signature type:** k‑hop degree distributions or random‑walk probability profiles.
        * **Similarity metric:** Cosine or correlation similarity.
        * **Clustering algorithm:** Spectral clustering or hierarchical clustering.
        * **Number of clusters:** Let the DSS choose automatically or set a value manually.
        
        The page displays a similarity heatmap, a network coloured by roles and a
        summary of each role in terms of average centrality measures and size.  A
        node selector in the sidebar lets you highlight specific nodes on the
        role plot; the selected nodes’ role assignments and centrality values
        are shown below the plot.  You can also compare role clusters to
        communities using Adjusted Rand Index (ARI) and Normalised Mutual
        Information (NMI).
        
        ## Community Detection & Robustness
        
        Community detection algorithms partition the network into highly connected
        subgroups.  The DSS implements Louvain, Girvan–Newman and spectral
        clustering methods.  For each method the modularity `Q` score and
        cluster statistics are reported.  You can examine robustness by
        repeatedly removing a fraction of edges and observing how the community
        assignments change (ARI) and how modularity drops.
        
        Comparing community partitions with role clusters helps understand whether
        structural roles align with densely connected factions.  As with the
        centrality and roles pages, you can select nodes to highlight them on
        the community‑coloured plot and view their community and role labels
        alongside their centrality measures.
        
        ## Kemeny Analysis
        
        The Kemeny constant measures the expected time to go from one random node
        to another in a Markov chain defined on the network【429076285098930†L81-L94】.  Smaller values indicate
        faster mixing and better overall connectivity.  On this page you can:
        
        * View the baseline Kemeny constant for the entire network.
        * Interactively remove nodes (by selecting them in a list) and observe how
          the Kemeny constant changes.  The baseline value (no removals) is
          always plotted at the zero point of the horizontal axis; points at
          1, 2, … show the Kemeny constant after each successive removal.
          A decrease after removing a node suggests that the node was hindering
          connectivity (e.g. a bottleneck).
        * A network plot shows the current graph; removed nodes are outlined in
          red **and** rendered with a different marker shape (an “×”).  All
          node identifiers are displayed directly on the plot so that you can
          easily see which nodes have been removed.  Hovering over a node
          reveals whether it has been removed.
        * Choose whether to recompute the constant on the largest connected
          component when removals disconnect the graph.
        
        ## Arrest Optimisation
        
        The agency has two departments to arrest members of the network.  To
        maximise arrests and minimise warnings (information leaks), it is
        preferable that connected members are assigned to the same department and
        that department capacities (ceil(`N`/2)) are respected【659313343491487†L195-L250】.  This page formulates the
        problem as a balanced cut optimisation.  You can:
        
        * **Select a community detection method** to determine which edges are
          penalised more heavily when cut.
        * **Adjust the regret strength (alpha):** Higher values penalise
          splitting edges within the same community and splitting high‑centrality
          nodes across departments.
        * **Adjust the penalty strength (beta):** Determines how many arrests
          are lost for each warning (cross‑department edge).
        * **Choose a centrality metric** to weight high‑centrality nodes in the
          regret term.
        
        The page displays the resulting assignment (department 0 or 1) on the
        network, the objective value, the number of cross‑department edges and
        the estimated number of effective arrests.  If an integer linear
        programming solver is unavailable, the DSS falls back to a heuristic.
        
        ## Recommended Workflow
        
        1. **Upload your network** on the first page and review its basic
           properties.  Resolve any data issues indicated by warnings.
        2. **Analyse centrality** to identify key players and potential
           influencers.  Adjust weighting schemes to see how rankings change.
        3. **Examine structural roles** to understand functional positions
           (brokers, hubs, peripherals) that may not align with centrality alone.
        4. **Detect communities** and evaluate robustness to see whether the
           network splits into stable factions.  Compare these with roles.
        5. **Assess connectivity** with the Kemeny constant and identify nodes
           whose removal improves mixing (possible targets for disruption).
        6. **Optimise arrests** by assigning members to departments, balancing
           capacity and minimising warnings.
        
        ## Glossary
        
        * **Centrality:** Quantitative measure of node importance in a network.
        * **Katz centrality:** Centrality measure incorporating paths of all
          lengths, attenuated by a factor of `alpha` per step.
        * **Kemeny constant:** Sum of mean first passage times; reflects network
          mixing speed.
        * **Modularity (Q):** Quality of a partition; higher values indicate
          dense intra‑community and sparse inter‑community connections.
        * **Adjusted Rand Index (ARI):** Metric to compare two partitions; 1
          indicates identical partitions, 0 indicates random agreement.
        * **Normalised Mutual Information (NMI):** Normalised measure of shared
          information between two partitions; ranges from 0 to 1.
        * **Balanced cut:** Graph partition problem with capacity constraints.
        
        ## Limitations
        
        * The results depend on the quality and completeness of the network data.
        * Community detection heuristics may yield different partitions on
          repeated runs; robustness analysis helps gauge stability.
        * The ILP solver used in the arrest optimisation may time out for very
          large networks; a heuristic solution is provided as a fallback.
        """
    )


if __name__ == "__main__":
    page()