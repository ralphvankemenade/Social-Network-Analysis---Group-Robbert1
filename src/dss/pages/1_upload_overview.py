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
#                 ### Welcome
                
#                 This dashboard allows you to analyze network structures using advanced graph-based methods.  
#                 Everything starts by uploading a network file.
                
#                 ---
                
#                 ### What to upload
                
#                 Upload a **Matrix Market (.mtx)** file that represents a network adjacency matrix.
                
#                 - Each row and column corresponds to a node  
#                 - Non-zero values indicate connections between nodes  
                
#                 ---
                
#                 ### What happens next
                
#                 After uploading the file:
                
#                 - The network is automatically loaded and validated  
#                 - All analysis pages in the navigation menu become available  
#                 - Centrality, roles, communities, robustness, and optimization analyses can be performed  
                
#                 ---
                
#                 ### No file uploaded yet
                
#                 If no file is uploaded, the analysis pages remain inactive.  
#                 Once a valid `.mtx` file is provided, you can immediately continue with the next steps.
                
#                 ---
                
#                 ### Tip
                
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
#             G = build_graph(adjacency, directed=False)
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
computes basic statistics. It also displays a baseline network plot
with no particular sizing or colouring. Once the graph is loaded,
subsequent pages can access it via `st.session_state`.

Additionally, this page stores the uploaded .mtx (filename + bytes) in
session state so the graph can be restored without re-uploading, as long
as the Streamlit session remains alive.
"""

from __future__ import annotations

from io import BytesIO
from typing import Optional

import streamlit as st

from dss.ui.state import (
    init_state,
    set_state,
    get_state,
    clear_network_state,
    clear_stored_file,
    reset_all_results,
)
from dss.utils.io_mtx import load_mtx
from dss.graph.build_graph import build_graph
from dss.graph.stats import basic_statistics
from dss.utils.validation import validate_graph
from dss.ui.components import display_network


UPLOAD_WIDGET_KEY = "upload_overview_mtx_uploader"


def _load_build_store_from_filelike(filelike: BytesIO, filename: Optional[str]) -> None:
    """Load adjacency from a file-like, build graph, validate, and store in session state."""
    adjacency = load_mtx(filelike)
    G = build_graph(adjacency, directed=False)
    stats = validate_graph(G)

    # Store core objects
    set_state("graph", G)
    set_state("adjacency", adjacency)

    # Any analysis caches depend on the graph, so reset them
    reset_all_results()

    # Store warnings/validation info for immediate display (optional)
    set_state("_upload_validation", stats)


def _render_summary_and_plot() -> None:
    """Render summary metrics and the base network plot if a graph is loaded."""
    G = get_state("graph")
    if G is None:
        st.info("Please upload a `.mtx` file to begin the analysis.")
        return

    stats = get_state("_upload_validation") or validate_graph(G)
    basic = basic_statistics(G)

    st.subheader("Network Summary")
    cols = st.columns(4)
    cols[0].metric("Nodes", basic["N"])
    cols[1].metric("Edges", basic["E"])
    cols[2].metric("Density", f"{basic['density']:.3f}")
    cols[3].metric("Components", basic["components"])

    if not stats.get("symmetric", True):
        st.warning("Adjacency matrix is not symmetric. Edges may be directed.")
    if stats.get("self_loops", False):
        st.warning("Graph contains self loops. They have been removed.")
    if not stats.get("connected", True):
        st.info(
            "Graph is not connected. Analyses will operate on the entire graph but some "
            "metrics (e.g. Kemeny) will use the largest component."
        )

    st.subheader("Network Graph")
    col_left, _col_right = st.columns([3, 2], gap="large")
    with col_left:
        display_network(G, title="Base network")


def page() -> None:
    st.set_page_config(page_title="Upload & Overview", layout="wide")
    st.title("Upload & Overview")

    with st.expander("Quick User Guide", expanded=False):
        st.markdown(
            """
            ### Welcome

            This dashboard allows you to analyze network structures using advanced graph-based methods.  
            Everything starts by uploading a network file.

            ---

            ### What to upload

            Upload a **Matrix Market (.mtx)** file that represents a network adjacency matrix.

            - Each row and column corresponds to a node  
            - Non-zero values indicate connections between nodes  

            ---

            ### What happens next

            After uploading the file:

            - The network is automatically loaded and validated  
            - All analysis pages in the navigation menu become available  
            - Centrality, roles, communities, robustness, and optimization analyses can be performed  

            ---

            ### No file uploaded yet

            If no file is uploaded, the analysis pages remain inactive.  
            Once a valid `.mtx` file is provided, you can immediately continue with the next steps.

            ---

            ### Tip

            If you are unsure about the file format or interpretation of results, consult the **User Manual**
            in the navigation menu for detailed explanations and examples.
            """
        )

    init_state()

    # Controls above uploader (optional, but keeps your existing UX)
    stored_name = get_state("mtx_filename")
    stored_bytes = get_state("mtx_bytes")
    have_stored = bool(stored_name and stored_bytes)

    if have_stored:
        col_a, col_b = st.columns([3, 1], gap="large")
        with col_a:
            st.success(f"Stored network file: {stored_name}")
        with col_b:
            if st.button("Clear stored file", use_container_width=True):
                # Clear everything including widget value
                clear_network_state()
                clear_stored_file()

                # Clearing the uploader requires setting its widget state to None
                # This is allowed as long as we do it in response to a widget action.
                st.session_state[UPLOAD_WIDGET_KEY] = None

                st.rerun()

    uploaded_file = st.file_uploader(
        "Upload a Matrix Market (.mtx) file",
        type=["mtx"],
        key=UPLOAD_WIDGET_KEY,
        help="""
What is a .mtx file?
Upload a Matrix Market (.mtx) file representing a network adjacency matrix.

Rows and columns correspond to nodes; non-zero entries indicate connections.

After upload, the graph will be loaded and all analysis pages become available.
""",
    )

    # If a new file is uploaded, always prefer it and overwrite the stored file
    if uploaded_file is not None:
        try:
            file_bytes = uploaded_file.getvalue()
            filename = uploaded_file.name

            # Store the file itself in session state for later restore
            set_state("mtx_filename", filename)
            set_state("mtx_bytes", file_bytes)

            # Load and build graph from these bytes
            _load_build_store_from_filelike(BytesIO(file_bytes), filename)

        except Exception as e:
            st.error(f"Failed to load network: {e}")

    # If no uploader file, but we do have stored bytes, offer restore
    # Important: we do NOT try to fill the uploader programmatically.
    if uploaded_file is None and have_stored:
        col_r1, col_r2 = st.columns([3, 1], gap="large")
        with col_r1:
            st.info("No file selected in the uploader. You can restore the last stored network.")
        with col_r2:
            if st.button("Restore stored file", use_container_width=True):
                try:
                    _load_build_store_from_filelike(BytesIO(stored_bytes), stored_name)
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to restore stored network: {e}")

    _render_summary_and_plot()


if __name__ == "__main__":
    page()
