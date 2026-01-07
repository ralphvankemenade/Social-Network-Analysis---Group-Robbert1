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




#========================================================================================

# # File: src/dss/pages/5_kemeny_interactive.py
# """Streamlit page: Kemeny constant analysis with interactive EDGE removal."""

# from __future__ import annotations

# from typing import Any, List, Tuple

# import streamlit as st
# import matplotlib.pyplot as plt
# import networkx as nx

# from dss.ui.state import init_state, get_state
# from dss.ui.components import display_network
# from dss.analytics.kemeny import kemeny_constant, interactive_kemeny_edges, Edge

# def _edge_label(G: nx.Graph, e: Edge) -> str:
#     u, v = e
#     if G.is_directed():
#         return f"{u} -> {v}"
#     return f"{u} - {v}"

# def page() -> None:
#     st.set_page_config(page_title="Kemeny Analysis", layout="wide")
#     st.title("Kemeny Constant and Connectivity Analysis")

#     init_state()
#     G = get_state("graph")
#     if G is None:
#         st.info("No graph loaded. Please upload a `.mtx` file on the Upload page.")
#         return

#     # Compute baseline Kemeny constant
#     base_k = kemeny_constant(G)
#     st.metric("Kemeny constant (baseline)", f"{base_k:.3f}")

#     st.subheader("Remove edges and observe effect on Kemeny")

#     recompute_on_largest = st.checkbox(
#         "Recompute on largest component if disconnected",
#         value=True
#     )

#     edges: List[Edge] = list(G.edges())
#     edge_labels = [_edge_label(G, e) for e in edges]

#     # Keep ordering stable and readable
#     edges_sorted = [e for _, e in sorted(zip(edge_labels, edges), key=lambda x: x[0])]
#     labels_sorted = sorted(edge_labels)

#     # Multiselect returns labels, we map them back to edges by index
#     # so the order shown is stable. The user order in multiselect is not guaranteed,
#     # so we provide an explicit "removal order" editor below.
#     selected_labels = st.multiselect("Select edges to remove", labels_sorted, [])

#     # Build selected edges in the stable sort order
#     label_to_edge = {_edge_label(G, e): e for e in edges_sorted}
#     selected_edges_initial: List[Edge] = [label_to_edge[lbl] for lbl in selected_labels if lbl in label_to_edge]

#     st.caption("Removal order matters. Edit the order below if needed.")

#     # Editable order list
#     ordered_edges: List[Edge] = []
#     for i, e in enumerate(selected_edges_initial):
#         ordered_edges.append(e)

#     # Simple order controls
#     if ordered_edges:
#         col1, col2 = st.columns(2)
#         with col1:
#             st.write("Current removal order:")
#             st.write([_edge_label(G, e) for e in ordered_edges])
#         with col2:
#             st.write("Quick reorder (swap two positions):")
#             idx_a = st.number_input("Index A", min_value=0, max_value=len(ordered_edges) - 1, value=0, step=1)
#             idx_b = st.number_input("Index B", min_value=0, max_value=len(ordered_edges) - 1, value=len(ordered_edges) - 1, step=1)
#             if st.button("Swap A and B"):
#                 ordered_edges[idx_a], ordered_edges[idx_b] = ordered_edges[idx_b], ordered_edges[idx_a]

#     # Compute and display the Kemeny constant after removals
#     if ordered_edges:
#         result = interactive_kemeny_edges(G, ordered_edges, recompute_on_largest)

#         if result.kemeny == result.kemeny:  # not NaN
#             st.metric("Kemeny constant after removals", f"{result.kemeny:.3f}")
#         else:
#             st.warning("Kemeny constant is undefined for the selected removals.")

#         # Plot history including baseline at x=0
#         st.subheader("Kemeny constant after each removal")
#         fig, ax = plt.subplots()

#         kemeny_series = [base_k] + result.history
#         x_vals = list(range(0, len(kemeny_series)))

#         ax.plot(x_vals, kemeny_series, marker="o")
#         ax.set_xlabel("Number of removed edges")
#         ax.set_ylabel("Kemeny constant")
#         ax.grid()
#         ax.set_title("Kemeny constant versus number of removed edges")
#         st.pyplot(fig)

#         # Show network after removing edges
#         st.subheader("Network view (after removing edges)")
#         H = G.copy()
#         for u, v in ordered_edges:
#             if H.has_edge(u, v):
#                 H.remove_edge(u, v)
#             elif (not H.is_directed()) and H.has_edge(v, u):
#                 H.remove_edge(v, u)

#         st.caption("Removed edges:")
#         st.write([_edge_label(G, e) for e in ordered_edges])

#         display_network(
#             H,
#             node_size=None,
#             node_color=None,
#             highlight=[],
#             title="Graph after edge removals",
#             show_labels=True,
#         )
#     else:
#         st.info("Select edges from the list above to remove them and recompute the Kemeny constant.")


# if __name__ == "__main__":
#     page()



#========================================================================================
# File: src/dss/pages/5_kemeny_interactive.py
"""Streamlit page: Kemeny constant analysis with interactive EDGE removal."""

from __future__ import annotations

from typing import Dict, List

import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from dss.ui.state import init_state, get_state
from dss.ui.components import display_network
from dss.analytics.kemeny import kemeny_constant, interactive_kemeny_edges, Edge


def _edge_label(G: nx.Graph, e: Edge) -> str:
    u, v = e
    return f"{u} -> {v}" if G.is_directed() else f"{u} - {v}"


def _build_label_to_edge(G: nx.Graph) -> Dict[str, Edge]:
    edges: List[Edge] = list(G.edges())
    labels = [_edge_label(G, e) for e in edges]
    pairs = sorted(zip(labels, edges), key=lambda x: x[0])
    return {lbl: e for lbl, e in pairs}


def _sync_order(selected: List[str], label_to_edge: Dict[str, Edge]) -> List[str]:
    if "kemeny_edge_order" not in st.session_state:
        st.session_state["kemeny_edge_order"] = []

    order: List[str] = list(st.session_state["kemeny_edge_order"])
    selected_set = set(selected)

    # Remove unselected
    order = [lbl for lbl in order if lbl in selected_set]

    # Append new selections (deterministic)
    for lbl in sorted(selected_set):
        if lbl not in order and lbl in label_to_edge:
            order.append(lbl)

    st.session_state["kemeny_edge_order"] = order

    # If active not set yet, set before widget instantiation
    if "kemeny_edge_active" not in st.session_state:
        st.session_state["kemeny_edge_active"] = order[0] if order else None

    # If active disappeared, set before widget instantiation (safe here)
    if st.session_state.get("kemeny_edge_active") not in order:
        st.session_state["kemeny_edge_active"] = order[0] if order else None

    return order


def _move(order_key: str, label: str, direction: int) -> None:
    """direction: -1 up, +1 down. Only updates the order, not the widget value."""
    order: List[str] = list(st.session_state.get(order_key, []))
    if label not in order:
        return
    i = order.index(label)
    j = i + direction
    if j < 0 or j >= len(order):
        return
    order[i], order[j] = order[j], order[i]
    st.session_state[order_key] = order


def page() -> None:
    st.set_page_config(page_title="Kemeny Analysis", layout="wide")
    st.title("Kemeny Constant and Connectivity Analysis")

    init_state()
    G = get_state("graph")
    if G is None:
        st.info("No graph loaded. Please upload a `.mtx` file on the Upload page.")
        return

    base_k = kemeny_constant(G)
    st.metric("Kemeny constant (baseline)", f"{base_k:.3f}")

    st.subheader("Remove edges and observe effect on Kemeny")
    recompute_on_largest = st.checkbox(
        "Recompute on largest component if disconnected",
        value=True,
    )

    label_to_edge = _build_label_to_edge(G)
    all_labels = list(label_to_edge.keys())

    selected = st.multiselect(
        "Select edges to remove",
        options=all_labels,
        default=st.session_state.get("kemeny_edge_selected", []),
        key="kemeny_edge_selected",
    )

    order = _sync_order(selected, label_to_edge)

    st.markdown("### Removal order")
    if not order:
        st.info("Select edges above to start building a removal order.")
        return

    # Step table: baseline row 0 + removal rows 1..k
    order_df = pd.DataFrame({"Step": list(range(1, len(order) + 1)), "Edge removed": order})
    baseline_df = pd.DataFrame({"Step": [0], "Edge removed": ["Baseline (no removal)"]})
    order_df = pd.concat([baseline_df, order_df], ignore_index=True)

    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.dataframe(order_df, use_container_width=True, hide_index=True)

    with col_b:
        active = st.selectbox(
            "Edge to reorder",
            options=order,
            key="kemeny_edge_active",
        )

        b1, b2 = st.columns(2)
        with b1:
            if st.button("Up", use_container_width=True):
                _move("kemeny_edge_order", active, -1)
                st.rerun()
        with b2:
            if st.button("Down", use_container_width=True):
                _move("kemeny_edge_order", active, +1)
                st.rerun()

        if st.button("Remove", use_container_width=True):
            st.session_state["kemeny_edge_order"] = [lbl for lbl in order if lbl != active]
            st.session_state["kemeny_edge_selected"] = [lbl for lbl in selected if lbl != active]
            # set active before rerun (safe: widget will be re-created next run)
            new_order = st.session_state["kemeny_edge_order"]
            st.session_state["kemeny_edge_active"] = new_order[0] if new_order else None
            st.rerun()

    ordered_edges: List[Edge] = [label_to_edge[lbl] for lbl in st.session_state["kemeny_edge_order"]]

    result = interactive_kemeny_edges(G, ordered_edges, recompute_on_largest)
    if result.kemeny == result.kemeny:
        st.metric("Kemeny constant after removals", f"{result.kemeny:.3f}")
    else:
        st.warning("Kemeny constant is undefined for the selected removals.")

    st.subheader("Kemeny constant after each removal")
    fig, ax = plt.subplots()
    series = [base_k] + result.history
    ax.plot(list(range(len(series))), series, marker="o")
    ax.set_xlabel("Step")
    ax.set_ylabel("Kemeny constant")
    ax.grid()
    ax.set_title("Kemeny constant versus removal steps")
    st.pyplot(fig)

    st.subheader("Network view (after removing edges)")
    H = G.copy()
    for u, v in ordered_edges:
        if H.has_edge(u, v):
            H.remove_edge(u, v)
        elif (not H.is_directed()) and H.has_edge(v, u):
            H.remove_edge(v, u)

    display_network(
        H,
        node_size=None,
        node_color=None,
        highlight=[],
        title="Graph after edge removals",
        show_labels=True,
    )


if __name__ == "__main__":
    page()
