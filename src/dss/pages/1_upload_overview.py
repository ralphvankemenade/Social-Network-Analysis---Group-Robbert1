"""Streamlit page: Upload & Overview.

This page allows the user to upload a `.mtx` file, validates it and
computes basic statistics.  It also displays a baseline network plot
with no particular sizing or colouring.  Once the graph is loaded,
subsequent pages can access it via `st.session_state`.
"""

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


def page() -> None:
    st.set_page_config(page_title="Upload & Overview", layout="wide")
    st.title("Upload & Overview")

    with st.expander("Quick User Guide", expanded=False):
        st.markdown(
            """
         
            """,
        )
    # Initialise session state
    init_state()
    # File uploader
    uploaded_file = st.file_uploader("Upload a Matrix Market (.mtx) file", type=["mtx"], help="TESTTETSTSTSTSTS")
    if uploaded_file is not None:
        try:
            # Load adjacency matrix
            adjacency = load_mtx(uploaded_file)
            # st.write(adjacency)
            # Build graph (assume undirected by default)
            G = build_graph(adjacency, directed=False)
            # Validate graph
            stats = validate_graph(G)
            # Store in session state
            set_state("graph", G)
            set_state("adjacency", adjacency)
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
                st.info("Graph is not connected.  Analyses will operate on the entire graph but some metrics (e.g. Kemeny) will use the largest component.")
            # Plot the graph
            st.subheader("Network Graph")
            display_network(G, title="Base network")
        except Exception as e:
            st.error(f"Failed to load network: {e}")
    else:
        st.info("Please upload a `.mtx` file to begin the analysis.")


if __name__ == "__main__":
    page()
