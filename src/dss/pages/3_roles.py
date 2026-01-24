# """Streamlit page: Role identification via similarity clustering."""

# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
# import matplotlib.patches as mpatches
# import matplotlib.pyplot as plt
# import matplotlib.lines as mlines

# from dss.ui.state import init_state, get_state, set_state
# from dss.analytics.roles import compute_roles, leaderranking
# from dss.analytics.centrality import compute_centralities
# from dss.ui.components import display_network, display_heatmap
# from dss.analytics.communities import compute_communities


# def page() -> None:
#     st.set_page_config(page_title="Role Analysis", layout="wide")
#     st.title("Role Identification via Similarity Clustering")

#     init_state()
#     G = get_state("graph")
#     if G is None:
#         st.info("No graph loaded.  Please upload a `.mtx` file on the Upload page.")
#         return

#     with st.expander("Quick User Guide", expanded=False):
#         st.markdown(
#             """
         
#             """,
#         )

    
#         col_left, col_right = st.columns([3, 2], gap="large")
    
#         with col_left:
#             st.markdown("#### Usable methods for role identification")
#             st.markdown(
#                 """
#                 **Cooper and Barahona** \\
#                 computes a similarity matrix by comparing nodes'
#                 flow profiles, which are paths of various lengths from one node
#                 to another, where these flow profiles can be created using either k-hop
#                 or random walk signatures. For the comparison, we provide both cosine and
#                 correlation distance functions. After computing the similarity matrix,
#                 we can cluster the nodes and assign nodes their respective roles
#                 through this clustering.

                
#                 **RoleSim** \\
#                 computes a similarity matrix through a iterative procedure, where
#                 it uses information on the a specific set of neighbours in the neighbourhoods 
#                 for each node combination, even if nodes are not connected in the network itself.
#                 After obtaining the similarity matrix, clustering is performed to assign roles. 
#                 Pre-specified value for the decay factor can be adjusted to fit the user's needs.

#                 **RoleSim*** \\
#                 takes a similar approach as RoleSim, with the difference being that 
#                 RoleSim* uses information of the entire neighbourhood of the nodes that are being compared.
#                 Again, after construction of the similarity matrix, through the iterative RoleSim* 
#                 procedure, clustering is performed to assign roles.
#                 Pre-specified value for the decay factor and weight balancing factor can be adjusted to 
#                 fit the user's needs.

#                 **RolX** \\
#                 takes a different approach to the other methods, as no similarity
#                 matrix is constructed in this instance. Instead, it creates a feature vector,
#                 which in our case includes a number of centrality measures, as well as some other information
#                 regarding the network. Then this feature vector is used in the RolX procedure to
#                 group nodes together, and assign roles to each of these groups.
#                 """
#             )
        
#         with col_right:
#             st.markdown("#### Clustering")
#             st.markdown(
#                 """
#                 All methods, excluding RolX, make use of the same clustering methods,
#                 which are either spectral or hierarchical clustering, to assign the roles to 
#                 each of the nodes. For all methods, the number of roles can be pre-specified, or 
#                 set to automatically compute the optimal number of roles in the network.
#                 """
#             )
#             st.markdown("#### Node inspection")
#             st.markdown(
#                 """
#                 Can select any number of nodes, which will then be highlighted on the graph
#                 to provide additional focus on a particular node, or set of ndoes.
#                 """

#             )
#             st.markdown("#### Computation")
#             st.markdown(
#                 """
#                 As some of the methods take longer to compute, we include a compute button that needs to be
#                 clicked to compute the role identification with new methods and/or parameters.
#                 """

#             )
#             st.markdown("#### Leader rankings")
#             st.markdown(
#                 """
#                 This part computes a score for each role, where a higher score indicates higher leadership value, thus members of this
#                 role being more likely to be a leader in the network.
#                 """

#             )

    
#     # Sidebar parameters
#     st.sidebar.header("Role identification methods")
#     method = st.sidebar.selectbox("Method", ["Cooper and Barahona", "RoleSim", "RoleSim*", "RolX"], index=0, help="Which method to use for the role identification")
#     info = {}

#     st.sidebar.header("Role similarity parameters")
#     if method == "Cooper and Barahona":
#         info["signature"] = st.sidebar.selectbox("Structural signature", ["k-hop", "random-walk"], index=0, help="Which method to use for creating flow profiles")
#         if info["signature"] == "k-hop":
#             info["k"] = st.sidebar.slider("Max hop (k)", 1, 5, 3, help="Maximum number of hops to be used for creating flow profiles")
#             info["t"] = 3
#         else:
#             info["t"] = st.sidebar.slider("Random-walk steps (t)", 1, 5, 3, help="Maximum number of steps to be used for creating flow profiles")
#             info["k"] = 3
#         info["similarity_metric"] = st.sidebar.selectbox("Similarity metric", ["cosine", "correlation"], index=0, help="Which metric to use for computing the distance between nodes' flow profiles")
    
#     if method == "RoleSim" or method == "RoleSim*":
#         info["beta"] = st.sidebar.slider("beta (decay factor)", 0., 1., 0.1, help="Higher values result in higher decay of previous RoleSim(*) scores, as well as a higher base RoleSim(*) score")
#         if method == "RoleSim*":
#             info["lambd"] = st.sidebar.slider("lambda (weight balancing factor)", 0., 1.,0.8, help="Higher values give more weight towards the set of information used in RoleSim, while lower values give more weight to combinations of neighbours not included in that set")
#         info["maxiter"] = st.sidebar.slider("Maximum number of iterations",5,1000,100, help="Maximum number of iterations to use. Lower values might result in lower computation time with the cost of some loss of accuracy. Use with caution")

#     st.sidebar.header("Role identification")
#     if method == "Cooper and Barahona" or method == "RoleSim" or method == "RoleSim*": 
#         info["clustering_method"] = st.sidebar.selectbox("Role identification method", ["spectral", "hierarchical"], index=0, help="Which clustering method to use")

#     auto_roles = st.sidebar.checkbox("Auto-detect number of roles", value=False, help="Automatically select number of roles")
#     if auto_roles:
#         info["n_roles"] = None
#     else:
#         if method == "RolX":
#             info["n_roles"] = st.sidebar.slider("Number of roles", 2, 6, 3, help="Set number of roles manually")
#         else:
#             info["n_roles"] = st.sidebar.slider("Number of roles", 2, max(2, int(np.ceil(np.sqrt(G.number_of_nodes())))), 3, help="Set number of roles manually")

#     '''
#     compute_button = st.sidebar.button("Compute roles")
#     if compute_button or get_state("role_result") is None:
#         # Compute centralities for summary statistics
#         centralities = compute_centralities(G)
#         role_result = compute_roles(
#             G,
#             method=method,
#             signature=signature,
#             k=k,
#             t=t,
#             similarity_metric=similarity_metric,
#             clustering_method=clustering_method,
#             n_clusters=n_roles,
#             centrality_table=centralities,
#         )
#         set_state("role_result", role_result)
#     else:
#         role_result = get_state("role_result")
#     '''
#     # compute_button = st.sidebar.button("Compute roles", help="Compute selected role identification method with specified parameters and clustering methods")
#     # if compute_button or get_state("role_result") is None:
#     #     with st.spinner("calculating..."):
#     #         # Compute centralities for summary statistics
#     #         centralities = compute_centralities(G)
#     #         role_result = compute_roles(
#     #             G,
#     #             method=method,
#     #             info=info,
#     #             centralities=centralities
#     #         )
#     #         set_state("role_result", role_result)
#     # else:
#     #     role_result = get_state("role_result")

#     compute_button = st.sidebar.button(
#         "Compute roles",
#         help="Compute selected role identification method with specified parameters and clustering methods",
#     )
    
#     if compute_button or get_state("role_result") is None:
#         progress = st.progress(0)
#         status = st.empty()
    
#         def progress_cb(p: float, msg: str) -> None:
#             # Clamp progress to [0, 1] and update UI
#             p = max(0.0, min(1.0, float(p)))
#             progress.progress(p)
#             status.write(msg)
    
#         try:
#             progress_cb(0.02, "Computing centralities")
#             # Compute centralities for summary statistics
#             centralities = compute_centralities(G)
    
#             progress_cb(0.08, "Computing roles")
#             role_result = compute_roles(
#                 G,
#                 method=method,
#                 info=info,
#                 centralities=centralities,
#                 progress_cb=progress_cb,
#             )
#             set_state("role_result", role_result)
    
#             progress_cb(1.00, "Finished")
#         finally:
#             # Optional: keep status text, but remove the bar if you prefer
#             # progress.empty()
#             pass
#     else:
#         role_result = get_state("role_result")


#     if method == "-":
#         st.text("RolX does not yet work in the current iteration of this DSS.")
#     else:
#         # Display similarity heatmap
#         #st.subheader("Role similarity heatmap",help="Heatmap with use of the similarity matrix. Can become hard to identify nodes when the number of nodes grow")
#         #if method == "RolX":
#         #    st.text('RolX does not compute similarity scores in such a manner that role similarity can be compared in the usual form')
#         #else:
#         #    display_heatmap(role_result.similarity_matrix, list(G.nodes()), caption="Role similarity")
#         # Display role summary
#         st.subheader("Role cluster summary", help="Summary of averages of the centrality statistics for each of the roles")
#         st.dataframe(role_result.summary)

#         col_left, col_right = st.columns([2, 3])
#         with col_left:
#             st.subheader("Leader rankings", help="Which roles are more likely to be leader/follower roles, where the higher the score, the more likely it is that the role to consists of leaders")
#             st.dataframe(leaderranking(role_result.summary))
#             # Colour map for roles
#             role_colors = {node: role_result.labels[node] for node in G.nodes()}
#         with col_right:
#             # Plot network coloured by roles with labels and interactive highlights
#             st.subheader("Network coloured by roles", help="Visual representation of the network, where each role has its own colour")
#             # Node selection for highlight and inspection
# #             st.sidebar.subheader("Select nodes to inspect")
# #             selected_nodes = st.sidebar.multiselect(
# #                 "Nodes", options=list(G.nodes()), default=[], help="""
# # Select one or more nodes to inspect in detail.

# # Selected nodes will:
# # - Always be highlighted in the network view
# # - Appear in a detailed table at the bottom of this page
# # """
# #                 )
#             selected_nodes = st.sidebar.multiselect(
#                     "Select nodes to inspect",
#                     options=list(G.nodes()),
#                     default=[],
#                     help="""
# Select one or more nodes to inspect in detail.

# Selected nodes will:
# - Always be highlighted in the network view
# - Appear in a detailed table at the bottom of this page
# """
#             )
#             # Highlight nodes that are selected
#             # highlight_nodes = selected_nodes
#             highlight_nodes_selected = list(selected_nodes)
#             # display_network(
#             #     G,
#             #     node_color=role_colors,
#             #     highlight=highlight_nodes,
#             #     title="Roles",
#             #     show_labels=True,
#             # )
#             display_network(
#                 G,
#                 node_color=role_colors,
#                 highlight_selected=highlight_nodes_selected,
#                 title="Roles",
#                 show_labels=True,
#             )
            
#         #role_colors = { 0: "#440154", 1: "#FDE725", 2: "#218855", 3: "#5B39C8", 4: "#BF1515", 5: "#1BACEE" }
#         #role_patches = [mpatches.Patch(color=color, label=f"Role {role + 1}") for role, color in role_colors.items()]

#         #legend_items = role_patches
#         #plt.legend(handles=legend_items, loc="upper right")
#         #st.pyplot(plt.gcf())
#         # Show details for selected nodes
#         if selected_nodes:
#             st.subheader("Selected node details", help="Details on highlighted node(s), like centrality values and role(s) of the node(s)")
#             # Build a DataFrame with role label and basic centrality measures
#             centralities = compute_centralities(G)
#             data = centralities.loc[selected_nodes].copy()
#             data["role_cluster"] = [role_result.labels[n] for n in selected_nodes]
#             st.dataframe(data)
    
            
#         # Compare roles to communities if available
#         st.subheader("Comparison with community clustering", help="Compare the results of the role identification with some of the results of the community clustering")
#         comm_method = st.selectbox("Community method for comparison", ["louvain", "girvan_newman", "spectral"], index=0)
#         # Compute community result (cached per method)
#         if get_state("community_results").get(comm_method) is None:
#             comm_result = compute_communities(G, method=comm_method, k=2)
#             get_state("community_results")[comm_method] = comm_result
#         comm_result = get_state("community_results")[comm_method]
#         # Compute ARI and NMI between role labels and community labels
#         role_labels_list = [role_result.labels[node] for node in G.nodes()]
#         comm_labels_list = [comm_result.labels[node] for node in G.nodes()]
#         ari = adjusted_rand_score(role_labels_list, comm_labels_list)
#         nmi = normalized_mutual_info_score(role_labels_list, comm_labels_list)
#         st.write(f"Adjusted Rand Index between roles and communities: {ari:.3f}")
#         st.write(f"Normalized Mutual Information: {nmi:.3f}")
#         # Confusion matrix
#         df_conf = pd.DataFrame(
#             {
#                 "role": role_labels_list,
#                 "community": comm_labels_list,
#             }
#         )
#         confusion = pd.crosstab(df_conf["role"], df_conf["community"])
#         st.dataframe(confusion)


# if __name__ == "__main__":
#     page()



"""Streamlit page: Role identification via similarity clustering."""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from dss.ui.state import init_state, get_state, set_state
from dss.analytics.roles import compute_roles, leaderranking
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

    with st.expander("Quick User Guide", expanded=False):
        st.markdown(
            """
         
            """,
        )

        col_left, col_right = st.columns([3, 2], gap="large")

        with col_left:
            st.markdown("#### Usable methods for role identification")
            st.markdown(
                """
                **Cooper and Barahona** \\
                computes a similarity matrix by comparing nodes'
                flow profiles, which are paths of various lengths from one node
                to another, where these flow profiles can be created using either k-hop
                or random walk signatures. For the comparison, we provide both cosine and
                correlation distance functions. After computing the similarity matrix,
                we can cluster the nodes and assign nodes their respective roles
                through this clustering.

                
                **RoleSim** \\
                computes a similarity matrix through a iterative procedure, where
                it uses information on the a specific set of neighbours in the neighbourhoods 
                for each node combination, even if nodes are not connected in the network itself.
                After obtaining the similarity matrix, clustering is performed to assign roles. 
                Pre-specified value for the decay factor can be adjusted to fit the user's needs.

                **RoleSim*** \\
                takes a similar approach as RoleSim, with the difference being that 
                RoleSim* uses information of the entire neighbourhood of the nodes that are being compared.
                Again, after construction of the similarity matrix, through the iterative RoleSim* 
                procedure, clustering is performed to assign roles.
                Pre-specified value for the decay factor and weight balancing factor can be adjusted to 
                fit the user's needs.

                **RolX** \\
                takes a different approach to the other methods, as no similarity
                matrix is constructed in this instance. Instead, it creates a feature vector,
                which in our case includes a number of centrality measures, as well as some other information
                regarding the network. Then this feature vector is used in the RolX procedure to
                group nodes together, and assign roles to each of these groups.
                """
            )

        with col_right:
            st.markdown("#### Clustering")
            st.markdown(
                """
                All methods, excluding RolX, make use of the same clustering methods,
                which are either spectral or hierarchical clustering, to assign the roles to 
                each of the nodes. For all methods, the number of roles can be pre-specified, or 
                set to automatically compute the optimal number of roles in the network.
                """
            )
            st.markdown("#### Node inspection")
            st.markdown(
                """
                Can select any number of nodes, which will then be highlighted on the graph
                to provide additional focus on a particular node, or set of ndoes.
                """
            )
            st.markdown("#### Computation")
            st.markdown(
                """
                As some of the methods take longer to compute, we include a compute button that needs to be
                clicked to compute the role identification with new methods and/or parameters.
                """
            )
            st.markdown("#### Leader rankings")
            st.markdown(
                """
                This part computes a score for each role, where a higher score indicates higher leadership value, thus members of this
                role being more likely to be a leader in the network.
                """
            )

    # Sidebar parameters
    st.sidebar.header("Role identification methods")
    method = st.sidebar.selectbox(
        "Method",
        ["Cooper and Barahona", "RoleSim", "RoleSim*", "RolX"],
        index=0,
        help="Which method to use for the role identification",
    )
    info = {}

    st.sidebar.header("Role similarity parameters")
    if method == "Cooper and Barahona":
        info["signature"] = st.sidebar.selectbox(
            "Structural signature",
            ["k-hop", "random-walk"],
            index=0,
            help="Which method to use for creating flow profiles",
        )
        if info["signature"] == "k-hop":
            info["k"] = st.sidebar.slider(
                "Max hop (k)",
                1,
                5,
                3,
                help="Maximum number of hops to be used for creating flow profiles",
            )
            info["t"] = 3
        else:
            info["t"] = st.sidebar.slider(
                "Random-walk steps (t)",
                1,
                5,
                3,
                help="Maximum number of steps to be used for creating flow profiles",
            )
            info["k"] = 3
        info["similarity_metric"] = st.sidebar.selectbox(
            "Similarity metric",
            ["cosine", "correlation"],
            index=0,
            help="Which metric to use for computing the distance between nodes' flow profiles",
        )

    if method == "RoleSim" or method == "RoleSim*":
        info["beta"] = st.sidebar.slider(
            "beta (decay factor)",
            0.0,
            1.0,
            0.1,
            help="Higher values result in higher decay of previous RoleSim(*) scores, as well as a higher base RoleSim(*) score",
        )
        if method == "RoleSim*":
            info["lambd"] = st.sidebar.slider(
                "lambda (weight balancing factor)",
                0.0,
                1.0,
                0.8,
                help="Higher values give more weight towards the set of information used in RoleSim, while lower values give more weight to combinations of neighbours not included in that set",
            )
        info["maxiter"] = st.sidebar.slider(
            "Maximum number of iterations",
            5,
            1000,
            100,
            help="Maximum number of iterations to use. Lower values might result in lower computation time with the cost of some loss of accuracy. Use with caution",
        )

    st.sidebar.header("Role identification")
    if method == "Cooper and Barahona" or method == "RoleSim" or method == "RoleSim*":
        info["clustering_method"] = st.sidebar.selectbox(
            "Role identification method",
            ["spectral", "hierarchical"],
            index=0,
            help="Which clustering method to use",
        )

    auto_roles = st.sidebar.checkbox(
        "Auto-detect number of roles", value=False, help="Automatically select number of roles"
    )
    if auto_roles:
        info["n_roles"] = None
    else:
        if method == "RolX":
            info["n_roles"] = st.sidebar.slider(
                "Number of roles", 2, int(np.min([6,G.number_of_nodes()-1])), 3, help="Set number of roles manually"
            )
        else:
            info["n_roles"] = st.sidebar.slider(
                "Number of roles",
                2,
                max(2, int(np.ceil(np.sqrt(G.number_of_nodes())))),
                3,
                help="Set number of roles manually",
            )

    '''
    compute_button = st.sidebar.button("Compute roles")
    if compute_button or get_state("role_result") is None:
        # Compute centralities for summary statistics
        centralities = compute_centralities(G)
        role_result = compute_roles(
            G,
            method=method,
            signature=signature,
            k=k,
            t=t,
            similarity_metric=similarity_metric,
            clustering_method=clustering_method,
            n_clusters=n_roles,
            centrality_table=centralities,
        )
        set_state("role_result", role_result)
    else:
        role_result = get_state("role_result")
    '''
    # compute_button = st.sidebar.button("Compute roles", help="Compute selected role identification method with specified parameters and clustering methods")
    # if compute_button or get_state("role_result") is None:
    #     with st.spinner("calculating..."):
    #         # Compute centralities for summary statistics
    #         centralities = compute_centralities(G)
    #         role_result = compute_roles(
    #             G,
    #             method=method,
    #             info=info,
    #             centralities=centralities
    #         )
    #         set_state("role_result", role_result)
    # else:
    #     role_result = get_state("role_result")

    compute_button = st.sidebar.button(
        "Compute roles",
        help="Compute selected role identification method with specified parameters and clustering methods",
    )

    if compute_button or get_state("role_result") is None:
        # Create placeholders so the progress UI can be shown only during computation
        progress_ph = st.empty()
        status_ph = st.empty()

        progress = None

        def progress_cb(p: float, msg: str) -> None:
            # Clamp progress to [0, 1] and update UI
            nonlocal progress
            p = max(0.0, min(1.0, float(p)))

            # Lazily create the progress bar only when we first receive progress updates
            if progress is None:
                progress = progress_ph.progress(0)

            progress.progress(p)
            status_ph.write(msg)

        try:
            progress_cb(0.02, "Computing centralities")
            # Compute centralities for summary statistics
            centralities = compute_centralities(G)

            progress_cb(0.08, "Computing roles")
            role_result = compute_roles(
                G,
                method=method,
                info=info,
                centralities=centralities,
                progress_cb=progress_cb,
            )
            set_state("role_result", role_result)

            progress_cb(1.00, "Finished")
        finally:
            # Remove progress UI after computation so it does not persist on page reload
            progress_ph.empty()
            status_ph.empty()
    else:
        role_result = get_state("role_result")

    if method == "-":
        st.text("RolX does not yet work in the current iteration of this DSS.")
    else:
        # Display similarity heatmap
        #st.subheader("Role similarity heatmap",help="Heatmap with use of the similarity matrix. Can become hard to identify nodes when the number of nodes grow")
        #if method == "RolX":
        #    st.text('RolX does not compute similarity scores in such a manner that role similarity can be compared in the usual form')
        #else:
        #    display_heatmap(role_result.similarity_matrix, list(G.nodes()), caption="Role similarity")
        # Display role summary
        st.subheader("Role cluster summary", help="Summary of averages of the centrality statistics for each of the roles")
        st.dataframe(role_result.summary)

        col_left, col_right = st.columns([2, 3])
        with col_left:
            st.subheader(
                "Leader rankings",
                help="Which roles are more likely to be leader/follower roles, where the higher the score, the more likely it is that the role to consists of leaders.",
            )
            st.dataframe(leaderranking(role_result.summary))
            # Colour map for roles
            role_colors = {node: role_result.labels[node] for node in G.nodes()}
        with col_right:
            # Plot network coloured by roles with labels and interactive highlights
            st.subheader(
                "Network coloured by roles",
                help="Visual representation of the network, where each role has its own colour. A darker colour means a lower role number, and vice versa for higher role numbers.",
            )
            # Node selection for highlight and inspection
#             st.sidebar.subheader("Select nodes to inspect")
#             selected_nodes = st.sidebar.multiselect(
#                 "Nodes", options=list(G.nodes()), default=[], help="""
# Select one or more nodes to inspect in detail.

# Selected nodes will:
# - Always be highlighted in the network view
# - Appear in a detailed table at the bottom of this page
# """
#                 )
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
            # Highlight nodes that are selected
            # highlight_nodes = selected_nodes
            highlight_nodes_selected = list(selected_nodes)
            # display_network(
            #     G,
            #     node_color=role_colors,
            #     highlight=highlight_nodes,
            #     title="Roles",
            #     show_labels=True,
            # )
            display_network(
                G,
                node_color=role_colors,
                highlight_selected=highlight_nodes_selected,
                title="Roles",
                show_labels=True,
            )

        #role_colors = { 0: "#440154", 1: "#FDE725", 2: "#218855", 3: "#5B39C8", 4: "#BF1515", 5: "#1BACEE" }
        #role_patches = [mpatches.Patch(color=color, label=f"Role {role + 1}") for role, color in role_colors.items()]

        #legend_items = role_patches
        #plt.legend(handles=legend_items, loc="upper right")
        #st.pyplot(plt.gcf())
        # Show details for selected nodes
        if selected_nodes:
            st.subheader(
                "Selected node details",
                help="Details on highlighted node(s), like centrality values and role(s) of the node(s)",
            )
            # Build a DataFrame with role label and basic centrality measures
            centralities = compute_centralities(G)
            data = centralities.loc[selected_nodes].copy()
            data["role_cluster"] = [role_result.labels[n] for n in selected_nodes]
            st.dataframe(data)

        # Compare roles to communities if available
        st.subheader(
            "Comparison with community clustering",
            help="Compare the results of the role identification with some of the results of the community clustering",
        )
        
        if G.is_directed():
                 comm_method = st.selectbox(
            "Community method for comparison", ["spectral", "girvan_newman"], index=0
                 )
        else:
                 comm_method = st.selectbox(
            "Community method for comparison", ["spectral","louvain", "girvan_newman"], index=0
                 )
             
        # comm_method = st.selectbox(
        #     "Community method for comparison", ["spectral","louvain", "girvan_newman"], index=0
        # )
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
        st.subheader(f"Table: Number of nodes out of certain community present within a role", help="How many nodes from specific role are part of specific community, with rows indicating role number, and column indicating community number")
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
