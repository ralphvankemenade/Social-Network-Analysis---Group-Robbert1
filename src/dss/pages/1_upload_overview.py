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
computes basic statistics.  It also displays a baseline network plot
with no particular sizing or colouring.  Once the graph is loaded,
subsequent pages can access it via `st.session_state`.
"""

from __future__ import annotations

import io
from typing import Optional

import streamlit as st
import pandas as pd

from dss.ui.state import init_state, set_state, get_state
from dss.utils.io_mtx import load_mtx
from dss.graph.build_graph import build_graph
from dss.graph.stats import basic_statistics
from dss.utils.validation import validate_graph
from dss.ui.components import display_network


# Keep one stable key for the uploader widget.
# Never assign to st.session_state[UPLOAD_WIDGET_KEY] yourself.
UPLOAD_WIDGET_KEY = "upload_overview_mtx_uploader"


def _load_and_store_mtx(file_like: io.BytesIO, filename: str) -> None:
    """Load an MTX file-like object, build graph, validate, and store results.

    This centralizes the logic so we can reuse it for:
    - a freshly uploaded file
    - a previously stored file restored from session state bytes
    """
    # Load adjacency matrix
    adjacency = load_mtx(file_like)

    # Build graph (assume undirected by default)
    G = build_graph(adjacency, directed=False)

    # Validate graph
    stats = validate_graph(G)

    # Store core objects so other pages can reuse them
    set_state("graph", G)
    set_state("adjacency", adjacency)

    # Store file metadata and raw bytes so we can restore later
    set_state("mtx_filename", filename)
    # NOTE: file_like might be consumed by load_mtx in some implementations.
    # We store bytes upstream when we call this helper.

    # Also store validation stats (optional, useful for UI)
    set_state("mtx_validation_stats", stats)

    # When a new graph is loaded, invalidate dependent cached results
    # so pages recompute under the new network.
    set_state("centrality_result", None)
    set_state("role_result", None)
    set_state("community_results", {})
    set_state("kemeny_result", None)
    set_state("arrest_result", None)


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

                If you are unsure about the file format or interpretation of results, consult the **User Manual** in the navigation menu for detailed explanations and examples.
            """
        )

    # Initialise session state
    init_state()

    # Ensure our extra keys exist (safe defaults)
    if get_state("mtx_filename") is None:
        set_state("mtx_filename", None)
    if get_state("mtx_bytes") is None:
        set_state("mtx_bytes", None)
    if get_state("mtx_validation_stats") is None:
        set_state("mtx_validation_stats", None)

    # Small status line if we have something stored
    stored_name: Optional[str] = get_state("mtx_filename")
    has_stored: bool = bool(stored_name and get_state("mtx_bytes") is not None)

    # Controls row (keeps UI compact, no big green banner required)
    if has_stored:
        c1, c2, c3 = st.columns([6, 2, 2])
        with c1:
            st.markdown(f"**Stored file:** `{stored_name}`")
        with c2:
            # Re-load from stored bytes (no touching the uploader widget value)
            if st.button("Use stored file", use_container_width=True):
                try:
                    raw = get_state("mtx_bytes")
                    file_like = io.BytesIO(raw)
                    _load_and_store_mtx(file_like, stored_name)
                    st.success("Stored file loaded.")
                except Exception as e:
                    st.error(f"Failed to load stored network: {e}")
        with c3:
            if st.button("Clear stored file", use_container_width=True):
                set_state("graph", None)
                set_state("adjacency", None)
                set_state("mtx_filename", None)
                set_state("mtx_bytes", None)
                set_state("mtx_validation_stats", None)

                # Also clear downstream results
                set_state("centrality_result", None)
                set_state("role_result", None)
                set_state("community_results", {})
                set_state("kemeny_result", None)
                set_state("arrest_result", None)

                st.success("Stored file cleared.")

    # File uploader (cannot be prefilled in Streamlit)
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

    # If user uploads a new file, store it and load it
    if uploaded_file is not None:
        try:
            # Read bytes once (this is what we can reliably persist)
            raw_bytes = uploaded_file.getvalue()
            filename = uploaded_file.name

            # Persist bytes + name for later restore
            set_state("mtx_bytes", raw_bytes)
            set_state("mtx_filename", filename)

            # Load from bytes
            file_like = io.BytesIO(raw_bytes)
            _load_and_store_mtx(file_like, filename)

        except Exception as e:
            st.error(f"Failed to load network: {e}")

    # From here on: if we have a graph in state, show the old summary + plot
    G = get_state("graph")
    adjacency = get_state("adjacency")

    if G is not None and adjacency is not None:
        # Show summary metrics (original behaviour)
        basic = basic_statistics(G)
        st.subheader("Network Summary")
        cols = st.columns(4)
        cols[0].metric("Nodes", basic["N"])
        cols[1].metric("Edges", basic["E"])
        cols[2].metric("Density", f"{basic['density']:.3f}")
        cols[3].metric("Components", basic["components"])

        # Symmetry and self-loop warnings (original behaviour)
        stats = get_state("mtx_validation_stats") or validate_graph(G)

        if not stats.get("symmetric", True):
            st.warning("Adjacency matrix is not symmetric.  Edges may be directed.")
        if stats.get("self_loops", False):
            st.warning("Graph contains self loops.  They have been removed.")
        if not stats.get("connected", True):
            st.info(
                "Graph is not connected.  Analyses will operate on the entire graph "
                "but some metrics (e.g. Kemeny) will use the largest component."
            )

        # Plot the graph (original behaviour)
        st.subheader("Network Graph")
        col_left, col_right = st.columns([3, 2], gap="large")
        with col_left:
            display_network(G, title="Base network")
    else:
        st.info("Please upload a `.mtx` file to begin the analysis.")


if __name__ == "__main__":
    page()

