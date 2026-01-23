# """Streamlit page: Upload & Overview.

# This page allows the user to upload a `.mtx` file, validates it and
# computes basic statistics.  It also displays a baseline network plot
# with no particular sizing or colouring.  Once the graph is loaded,
# subsequent pages can access it via `st.session_state`.
# """

# import streamlit as st
# import pandas as pd
# from dss.ui.state import init_state, set_state, get_state
# from dss.utils.io_mtx import load_mtx
# from dss.graph.build_graph import build_graph
# from dss.graph.stats import basic_statistics
# from dss.utils.validation import validate_graph
# from dss.utils.plotting import plot_network
# from dss.graph.layouts import compute_layout
# from dss.ui.components import display_network


# def page() -> None:
#     st.set_page_config(page_title="Upload & Overview", layout="wide")
#     st.title("Upload & Overview")

#     with st.expander("Quick User Guide", expanded=False):
#         st.markdown(
#             """
#                 #### Welcome
#                 This dashboard allows you to analyze network structures using advanced graph-based methods.  
#                 Everything starts by uploading a network file.
                
#                 #### What to upload
                
#                 Upload a **Matrix Market (.mtx)** file that represents a network adjacency matrix.
                
#                 - Each row and column corresponds to a node  
#                 - Non-zero values indicate connections between nodes  
               
#                 #### What happens next
                
#                 After uploading the file:
                
#                 - The network is automatically loaded and validated  
#                 - All analysis pages in the navigation menu become available  
#                 - Centrality, roles, communities, robustness, and optimization analyses can be performed  
                
#                 #### No file uploaded yet
                
#                 If no file is uploaded, the analysis pages remain inactive.  
#                 Once a valid `.mtx` file is provided, you can immediately continue with the next steps.
                
#                 #### Tip
                
#                 If you are unsure about the file format or interpretation of results, consult the **User Manual** in the navigation menu for detailed explanations and examples.
#             """
#         )

#     # Initialise session state
#     init_state()
#     # File uploader
#     # uploaded_file = st.file_uploader("Upload a Matrix Market (.mtx) file", type=["mtx"])
#     uploaded_file = st.file_uploader(
#         "Upload a Matrix Market (.mtx) file",
#         type=["mtx"],
#         help="""
#     What is a .mtx file?
#     Upload a Matrix Market (.mtx) file representing a network adjacency matrix.
    
#     Rows and columns correspond to nodes; non-zero entries indicate connections.
    
#     After upload, the graph will be loaded and all analysis pages become available.
#     """
#     )


#     if uploaded_file is not None:
#         try:
#             # Load adjacency matrix
#             adjacency = load_mtx(uploaded_file)
#             # st.write(adjacency)
#             # Build graph (assume undirected by default)
#             # G = build_graph(adjacency, directed=False)
#             G = build_graph(adjacency)
#             # Validate graph
#             stats = validate_graph(G)
#             # Store in session state
#             set_state("graph", G)
#             set_state("adjacency", adjacency)
#             # Show summary metrics
#             basic = basic_statistics(G)
#             st.subheader("Network Summary")
#             cols = st.columns(4)
#             cols[0].metric("Nodes", basic["N"])
#             cols[1].metric("Edges", basic["E"])
#             cols[2].metric("Density", f"{basic['density']:.3f}")
#             cols[3].metric("Components", basic["components"])
#             # Symmetry and self-loop warnings
#             if not stats["symmetric"]:
#                 st.warning("Adjacency matrix is not symmetric.  Edges may be directed.")
#             if stats["self_loops"]:
#                 st.warning("Graph contains self loops.  They have been removed.")
#             if not stats["connected"]:
#                 st.info("Graph is not connected.  Analyses will operate on the entire graph but some metrics (e.g. Kemeny) will use the largest component.")
#             # Plot the graph
#             st.subheader("Network Graph")
            
#             col_left, col_right = st.columns([3, 2], gap="large")
#             with col_left:
#                 display_network(G, title="Base network")
                
#         except Exception as e:
#             st.error(f"Failed to load network: {e}")
#     else:
#         st.info("Please upload a `.mtx` file to begin the analysis.")


# if __name__ == "__main__":
#     page()



"""Streamlit page: Upload & Overview.

This page allows the user to upload a `.mtx` file, validates it and
computes basic statistics.  It also displays a baseline network plot
with no particular sizing or colouring.  Once the graph is loaded,
subsequent pages can access it via `st.session_state`.
"""

import hashlib
import streamlit as st
import pandas as pd
from dss.ui.state import init_state, set_state, get_state
from dss.utils.io_mtx import load_mtx
from dss.graph.build_graph import build_graph
from dss.graph.stats import basic_statistics
from dss.utils.validation import validate_graph
from dss.utils.plotting import plot_network
from dss.graph.layouts import compute_layout
from dss.ui.components import display_network


def _uploaded_file_id(f) -> str:
    # Use name + size + content hash to uniquely identify uploads
    data = f.getvalue()
    h = hashlib.md5(data).hexdigest()
    return f"{f.name}__{f.size}__{h}"


def page() -> None:
    st.set_page_config(page_title="Upload & Overview", layout="wide")
    st.title("Upload & Overview")

    with st.expander("Quick User Guide", expanded=False):
        st.markdown(
            """
                #### Welcome
                This dashboard allows you to analyze network structures using advanced graph-based methods.  
                Everything starts by uploading a network file.
                
                #### What to upload
                
                Upload a **Matrix Market (.mtx)** file that represents a network adjacency matrix.
                
                - Each row and column corresponds to a node  
                - Non-zero values indicate connections between nodes  
               
                #### What happens next
                
                After uploading the file:
                
                - The network is automatically loaded and validated  
                - All analysis pages in the navigation menu become available  
                - Centrality, roles, communities, robustness, and optimization analyses can be performed  
                
                #### No file uploaded yet
                
                If no file is uploaded, the analysis pages remain inactive.  
                Once a valid `.mtx` file is provided, you can immediately continue with the next steps.
                
                #### Tip
                
                If you are unsure about the file format or interpretation of results, consult the **User Manual** in the navigation menu for detailed explanations and examples.
            """
        )

    # Initialise session state
    init_state()
    # File uploader
    # uploaded_file = st.file_uploader("Upload a Matrix Market (.mtx) file", type=["mtx"])
    uploaded_file = st.file_uploader(
        "Upload a Matrix Market (.mtx) file",
        type=["mtx"],
        help="""
    What is a .mtx file?
    Upload a Matrix Market (.mtx) file representing a network adjacency matrix.
    
    Rows and columns correspond to nodes; non-zero entries indicate connections.
    
    After upload, the graph will be loaded and all analysis pages become available.
    """,
    )

    if uploaded_file is not None:
        # If a new file is uploaded, reset all old graph-dependent state
        new_file_id = _uploaded_file_id(uploaded_file)
        old_file_id = st.session_state.get("current_file_id")

        if old_file_id != new_file_id:
            # Reset graph-dependent state keys
            reset_keys = [
                "graph",
                "adjacency",
                "layout",
                "centrality_result",
                "centrality_table",
                "centrality_method",
                "centrality_weights",
                "borda_weights",
                "roles_result",
                "roles_table",
                "communities_result",
                "communities_table",
                "kemeny_result",
                "removed_edges",
                "highlight_top",
                "highlight_selected",
                "highlight_arrested",
            ]

            for k in reset_keys:
                if k in st.session_state:
                    del st.session_state[k]

            # If you use Streamlit caches anywhere for graph computations, clear them as well
            try:
                st.cache_data.clear()
            except Exception:
                pass
            try:
                st.cache_resource.clear()
            except Exception:
                pass

            st.session_state["current_file_id"] = new_file_id

        try:
            # Load adjacency matrix
            adjacency = load_mtx(uploaded_file)
            # st.write(adjacency)
            # Build graph (assume undirected by default)
            # G = build_graph(adjacency, directed=False)
            G = build_graph(adjacency)
            # Validate graph
            stats = validate_graph(G)
            # Store in session state
            set_state("graph", G)
            set_state("adjacency", adjacency)
            set_state("layout", None)
            # Show summary metrics
            basic = basic_statistics(G)
            st.subheader("Network Summary")
            cols = st.columns(4)
            cols[0].metric("Nodes", basic["N"])
            cols[1].metric("Edges", basic["E"])
            cols[2].metric("Density", f"{basic['density']:.3f}")
            cols[3].metric("Components", basic["components"])
            # Symmetry and self-loop warnings
            if not stats["symmetric"]:
                st.warning("Adjacency matrix is not symmetric.  Edges may be directed.")
            if stats["self_loops"]:
                st.warning("Graph contains self loops.  They have been removed.")
            if not stats["connected"]:
                st.info(
                    "Graph is not connected.  Analyses will operate on the entire graph but some metrics (e.g. Kemeny) will use the largest component."
                )
            # Plot the graph
            st.subheader("Network Graph")

            col_left, col_right = st.columns([3, 2], gap="large")
            with col_left:
                display_network(G, title="Base network")

        except Exception as e:
            st.error(f"Failed to load network: {e}")
    else:
        st.info("Please upload a `.mtx` file to begin the analysis.")


if __name__ == "__main__":
    page()
