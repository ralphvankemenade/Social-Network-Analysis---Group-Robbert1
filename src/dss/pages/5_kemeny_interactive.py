# """Streamlit page: Kemeny constant analysis with interactive node removal."""

# import streamlit as st
# import matplotlib.pyplot as plt

# from dss.ui.state import init_state, get_state
# from dss.ui.components import display_network
# from dss.analytics.kemeny import kemeny_constant, interactive_kemeny


# def page() -> None:
#     st.set_page_config(page_title="Kemeny Analysis", layout="wide")
#     st.title("Kemeny Constant and Connectivity Analysis")
#     init_state()
#     G = get_state("graph")
#     if G is None:
#         st.info("No graph loaded.  Please upload a `.mtx` file on the Upload page.")
#         return
#     # Compute baseline Kemeny constant
#     base_k = kemeny_constant(G)
#     st.metric("Kemeny constant (baseline)", f"{base_k:.3f}")
#     # Provide an interactive removal interface
#     st.subheader("Remove nodes and observe effect on Kemeny")
#     nodes = list(G.nodes())
#     selected = st.multiselect("Select nodes to remove", nodes, [])
#     recompute_on_largest = st.checkbox("Recompute on largest component if disconnected", value=True)
#     # Compute and display the Kemeny constant after removals
#     if selected:
#         result = interactive_kemeny(G, selected, recompute_on_largest)
#         if result.kemeny == result.kemeny:  # not NaN
#             st.metric("Kemeny constant after removals", f"{result.kemeny:.3f}")
#         else:
#             st.warning("Kemeny constant is undefined for the selected removals.")
#         # Plot the history of Kemeny values as nodes are removed sequentially
#         st.subheader("Kemeny constant after each removal")
#         fig, ax = plt.subplots()
#         # x_vals = list(range(1, len(result.history) + 1))
#         # ax.plot(x_vals, result.history, marker="o")
#         # Include baseline as starting point
#         kemeny_series = [base_k] + result.history
#         x_vals = list(range(0, len(kemeny_series)))
        
#         ax.plot(x_vals, kemeny_series, marker="o")
#         # ax.axvline(0, linestyle="--", linewidth=1)
#         ax.set_xlabel("Number of removed nodes")
#         ax.set_ylabel("Kemeny constant")
#         # ax.set_ylim(bottom=max(kemeny_series) * 0.90)
#         ax.grid()
#         ax.set_title("Kemeny constant versus number of removed nodes")
#         st.pyplot(fig)
        
#         # Show the current network with removed nodes highlighted
#         st.subheader("Network view (removed nodes highlighted)")
#         display_network(
#             G,
#             node_size=None,
#             node_color=None,
#             highlight=selected,
#             title="Removed nodes are outlined in red",
#             show_labels=True,
#         )
#     else:
#         st.info("Select nodes from the list above to remove them and recompute the Kemeny constant.")
    
    
    


# if __name__ == "__main__":
#     page()






# File: src/dss/pages/5_kemeny_interactive.py
"""Streamlit page: Kemeny constant analysis with interactive EDGE removal."""

from __future__ import annotations

from typing import Any, List, Tuple

import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx

from dss.ui.state import init_state, get_state
from dss.ui.components import display_network
from dss.analytics.kemeny import kemeny_constant, interactive_kemeny_edges, Edge

def _edge_label(G: nx.Graph, e: Edge) -> str:
    u, v = e
    if G.is_directed():
        return f"{u} -> {v}"
    return f"{u} - {v}"

def page() -> None:
    st.set_page_config(page_title="Kemeny Analysis", layout="wide")
    st.title("Kemeny Constant and Connectivity Analysis")

    init_state()
    G = get_state("graph")
    if G is None:
        st.info("No graph loaded. Please upload a `.mtx` file on the Upload page.")
        return

    # Compute baseline Kemeny constant
    base_k = kemeny_constant(G)
    st.metric("Kemeny constant (baseline)", f"{base_k:.3f}")

    st.subheader("Remove edges and observe effect on Kemeny")

    recompute_on_largest = st.checkbox(
        "Recompute on largest component if disconnected",
        value=True
    )

    edges: List[Edge] = list(G.edges())
    edge_labels = [_edge_label(G, e) for e in edges]

    # Keep ordering stable and readable
    edges_sorted = [e for _, e in sorted(zip(edge_labels, edges), key=lambda x: x[0])]
    labels_sorted = sorted(edge_labels)

    # Multiselect returns labels, we map them back to edges by index
    # so the order shown is stable. The user order in multiselect is not guaranteed,
    # so we provide an explicit "removal order" editor below.
    selected_labels = st.multiselect("Select edges to remove", labels_sorted, [])

    # Build selected edges in the stable sort order
    label_to_edge = {_edge_label(G, e): e for e in edges_sorted}
    selected_edges_initial: List[Edge] = [label_to_edge[lbl] for lbl in selected_labels if lbl in label_to_edge]

    st.caption("Removal order matters. Edit the order below if needed.")

    # Editable order list
    ordered_edges: List[Edge] = []
    for i, e in enumerate(selected_edges_initial):
        ordered_edges.append(e)

    # Simple order controls
    if ordered_edges:
        col1, col2 = st.columns(2)
        with col1:
            st.write("Current removal order:")
            st.write([_edge_label(G, e) for e in ordered_edges])
        with col2:
            st.write("Quick reorder (swap two positions):")
            idx_a = st.number_input("Index A", min_value=0, max_value=len(ordered_edges) - 1, value=0, step=1)
            idx_b = st.number_input("Index B", min_value=0, max_value=len(ordered_edges) - 1, value=len(ordered_edges) - 1, step=1)
            if st.button("Swap A and B"):
                ordered_edges[idx_a], ordered_edges[idx_b] = ordered_edges[idx_b], ordered_edges[idx_a]

    # Compute and display the Kemeny constant after removals
    if ordered_edges:
        result = interactive_kemeny_edges(G, ordered_edges, recompute_on_largest)

        if result.kemeny == result.kemeny:  # not NaN
            st.metric("Kemeny constant after removals", f"{result.kemeny:.3f}")
        else:
            st.warning("Kemeny constant is undefined for the selected removals.")

        # Plot history including baseline at x=0
        st.subheader("Kemeny constant after each removal")
        fig, ax = plt.subplots()

        kemeny_series = [base_k] + result.history
        x_vals = list(range(0, len(kemeny_series)))

        ax.plot(x_vals, kemeny_series, marker="o")
        ax.set_xlabel("Number of removed edges")
        ax.set_ylabel("Kemeny constant")
        ax.grid()
        ax.set_title("Kemeny constant versus number of removed edges")
        st.pyplot(fig)

        # Show network after removing edges
        st.subheader("Network view (after removing edges)")
        H = G.copy()
        for u, v in ordered_edges:
            if H.has_edge(u, v):
                H.remove_edge(u, v)
            elif (not H.is_directed()) and H.has_edge(v, u):
                H.remove_edge(v, u)

        st.caption("Removed edges:")
        st.write([_edge_label(G, e) for e in ordered_edges])

        display_network(
            H,
            node_size=None,
            node_color=None,
            highlight=[],
            title="Graph after edge removals",
            show_labels=True,
        )
    else:
        st.info("Select edges from the list above to remove them and recompute the Kemeny constant.")


if __name__ == "__main__":
    page()
