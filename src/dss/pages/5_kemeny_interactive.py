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

# from typing import Dict, List

# import streamlit as st
# import matplotlib.pyplot as plt
# import networkx as nx
# import pandas as pd

# from dss.ui.state import init_state, get_state
# from dss.ui.components import display_network
# from dss.analytics.kemeny import kemeny_constant, interactive_kemeny_edges, Edge


# def _edge_label(G: nx.Graph, e: Edge) -> str:
#     u, v = e
#     su, sv = str(u), str(v)
#     return f"{su} -> {sv}" if G.is_directed() else f"{su} - {sv}"


# def _build_label_to_edge(G: nx.Graph) -> Dict[str, Edge]:
#     edges: List[Edge] = list(G.edges())
#     labels = [_edge_label(G, e) for e in edges]
#     pairs = sorted(zip(labels, edges), key=lambda x: x[0])
#     return {lbl: e for lbl, e in pairs}


# def _sync_order(selected: List[str], label_to_edge: Dict[str, Edge]) -> List[str]:
#     if "kemeny_edge_order" not in st.session_state:
#         st.session_state["kemeny_edge_order"] = []

#     order: List[str] = list(st.session_state["kemeny_edge_order"])
#     selected_set = set(selected)

#     order = [lbl for lbl in order if lbl in selected_set]
#     for lbl in sorted(selected_set):
#         if lbl not in order and lbl in label_to_edge:
#             order.append(lbl)

#     st.session_state["kemeny_edge_order"] = order
#     return order


# def _move_active(direction: int) -> None:
#     order: List[str] = list(st.session_state.get("kemeny_edge_order", []))
#     active: str = st.session_state.get("kemeny_edge_active", "")
#     if not order or active not in order:
#         return

#     i = order.index(active)
#     j = i + direction
#     if j < 0 or j >= len(order):
#         return

#     order[i], order[j] = order[j], order[i]
#     st.session_state["kemeny_edge_order"] = order

#     # keep the same active edge selected
#     st.session_state["kemeny_edge_active"] = active


# def _remove_active_edge() -> None:
#     order: List[str] = list(st.session_state.get("kemeny_edge_order", []))
#     selected: List[str] = list(st.session_state.get("kemeny_edge_selected_state", []))
#     active: str = st.session_state.get("kemeny_edge_active", "")

#     if not order or active not in order:
#         return

#     new_order = [lbl for lbl in order if lbl != active]
#     new_selected = [lbl for lbl in selected if lbl != active]

#     st.session_state["kemeny_edge_order"] = new_order
#     st.session_state["kemeny_edge_selected_state"] = new_selected

#     # Important: update the multiselect widget value in the callback
#     # (this runs before the next render, so Streamlit allows it)
#     st.session_state["kemeny_edge_selected_widget"] = list(new_selected)

#     # update active edge
#     if new_order:
#         st.session_state["kemeny_edge_active"] = new_order[0]
#     else:
#         st.session_state["kemeny_edge_active"] = ""


# def page() -> None:
#     st.set_page_config(page_title="Kemeny Analysis", layout="wide")
#     st.title("Kemeny Constant and Connectivity Analysis")

#     init_state()
#     G = get_state("graph")
#     if G is None:
#         st.info("No graph loaded. Please upload a `.mtx` file on the Upload page.")
#         return

#     base_k = kemeny_constant(G)
#     st.metric("Kemeny constant (baseline)", f"{base_k:.3f}")

#     st.subheader("Remove edges and observe effect on Kemeny")
#     recompute_on_largest = st.checkbox("Recompute on largest component if disconnected", value=True)

#     label_to_edge = _build_label_to_edge(G)
#     all_labels = list(label_to_edge.keys())

#     # Selected edges state (source of truth)
#     if "kemeny_edge_selected_state" not in st.session_state:
#         st.session_state["kemeny_edge_selected_state"] = []
#     if "kemeny_edge_selected_widget" not in st.session_state:
#         st.session_state["kemeny_edge_selected_widget"] = list(st.session_state["kemeny_edge_selected_state"])

#     selected_widget = st.multiselect(
#         "Select edges to remove",
#         options=all_labels,
#         default=st.session_state["kemeny_edge_selected_state"],
#         key="kemeny_edge_selected_widget",
#     )

#     # Persist widget -> state
#     st.session_state["kemeny_edge_selected_state"] = list(selected_widget)
#     selected = list(selected_widget)

#     order = _sync_order(selected, label_to_edge)

#     st.markdown("### Removal order")
#     if not order:
#         st.info("Select edges above to start building a removal order.")
#         return

#     order_df = pd.DataFrame({"Step": list(range(1, len(order) + 1)), "Edge removed": order})
#     order_df = pd.concat(
#         [pd.DataFrame({"Step": [0], "Edge removed": ["Baseline (no removal)"]}), order_df],
#         ignore_index=True,
#     )

#     col_a, col_b, _ = st.columns([1, 1, 3])
#     with col_a:
#         st.dataframe(order_df, use_container_width=True, hide_index=True)

#     with col_b:
#         # Active edge selection
#         if "kemeny_edge_active" not in st.session_state or st.session_state["kemeny_edge_active"] not in order:
#             st.session_state["kemeny_edge_active"] = order[0]

#         st.selectbox("Edge to reorder", options=order, key="kemeny_edge_active")

#         b1, b2 = st.columns(2)
#         with b1:
#             st.button("Up", use_container_width=True, on_click=_move_active, args=(-1,))
#         with b2:
#             st.button("Down", use_container_width=True, on_click=_move_active, args=(+1,))

#         st.button("Remove", use_container_width=True, on_click=_remove_active_edge)

#     ordered_edges: List[Edge] = [label_to_edge[lbl] for lbl in st.session_state["kemeny_edge_order"]]

#     result = interactive_kemeny_edges(G, ordered_edges, recompute_on_largest)
#     if result.kemeny == result.kemeny:
#         st.metric("Kemeny constant after removals", f"{result.kemeny:.3f}")
#     else:
#         st.warning("Kemeny constant is undefined for the selected removals.")

#     col_plot, col_graph = st.columns([1, 1])

#     with col_plot:
#         st.subheader("Kemeny constant after each removal")
#         fig, ax = plt.subplots()
#         series = [base_k] + result.history
#         ax.plot(list(range(len(series))), series, marker="o")
#         ax.set_xlabel("Step")
#         ax.set_ylabel("Kemeny constant")
#         ax.grid()
#         ax.set_title("Kemeny constant versus removal steps")
#         st.pyplot(fig)

#     with col_graph:
#         st.subheader("Network view (after removing edges)")

#         H = G.copy()
#         for u, v in ordered_edges:
#             if H.has_edge(u, v):
#                 H.remove_edge(u, v)
#             elif (not H.is_directed()) and H.has_edge(v, u):
#                 H.remove_edge(v, u)

#         display_network(
#             H,
#             node_size=None,
#             node_color=None,
#             highlight=[],
#             title="Graph after edge removals",
#             show_labels=True,
#             removed_edges=ordered_edges,
#         )


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
    su, sv = str(u), str(v)
    return f"{su} -> {sv}" if G.is_directed() else f"{su} - {sv}"


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

    order = [lbl for lbl in order if lbl in selected_set]
    for lbl in sorted(selected_set):
        if lbl not in order and lbl in label_to_edge:
            order.append(lbl)

    st.session_state["kemeny_edge_order"] = order
    return order


def _move_active(direction: int) -> None:
    order: List[str] = list(st.session_state.get("kemeny_edge_order", []))
    active: str = st.session_state.get("kemeny_edge_active", "")
    if not order or active not in order:
        return

    i = order.index(active)
    j = i + direction
    if j < 0 or j >= len(order):
        return

    order[i], order[j] = order[j], order[i]
    st.session_state["kemeny_edge_order"] = order
    st.session_state["kemeny_edge_active"] = active


def _remove_active_edge() -> None:
    order: List[str] = list(st.session_state.get("kemeny_edge_order", []))
    selected: List[str] = list(st.session_state.get("kemeny_edge_selected_state", []))
    active: str = st.session_state.get("kemeny_edge_active", "")

    if not order or active not in order:
        return

    new_order = [lbl for lbl in order if lbl != active]
    new_selected = [lbl for lbl in selected if lbl != active]

    st.session_state["kemeny_edge_order"] = new_order
    st.session_state["kemeny_edge_selected_state"] = new_selected

    # Safe: update widget value in callback
    st.session_state["kemeny_edge_selected_widget"] = list(new_selected)

    if new_order:
        st.session_state["kemeny_edge_active"] = new_order[0]
    else:
        st.session_state["kemeny_edge_active"] = ""


def page() -> None:
    st.set_page_config(page_title="Kemeny Analysis", layout="wide")
    st.title("Kemeny Constant and Connectivity Analysis")

    init_state()
    G = get_state("graph")
    if G is None:
        st.info("No graph loaded. Please upload a `.mtx` file on the Upload page.")
        return

    st.subheader("Remove edges and observe effect on Kemeny")
    recompute_on_largest = st.checkbox("Recompute on largest component if disconnected", value=True)

    label_to_edge = _build_label_to_edge(G)
    all_labels = list(label_to_edge.keys())

    # Source of truth for selection
    if "kemeny_edge_selected_state" not in st.session_state:
        st.session_state["kemeny_edge_selected_state"] = []
    if "kemeny_edge_selected_widget" not in st.session_state:
        st.session_state["kemeny_edge_selected_widget"] = list(st.session_state["kemeny_edge_selected_state"])

    selected_widget = st.multiselect(
        "Select edges to remove",
        options=all_labels,
        default=st.session_state["kemeny_edge_selected_state"],
        key="kemeny_edge_selected_widget",
    )

    st.session_state["kemeny_edge_selected_state"] = list(selected_widget)
    selected = list(selected_widget)

    order = _sync_order(selected, label_to_edge)
    ordered_edges: List[Edge] = [label_to_edge[lbl] for lbl in st.session_state.get("kemeny_edge_order", [])]

    base_k = kemeny_constant(G)
    result = interactive_kemeny_edges(G, ordered_edges, recompute_on_largest)

    # Kemeny interpretation:
    # Lower Kemeny means faster mixing / shorter expected travel time in the Markov chain,
    # so "better" here is LOWER.
    after_k = result.kemeny
    kemeny_defined = after_k == after_k  # not NaN
    delta = (after_k - base_k) if kemeny_defined else None

    # === MAIN CONTENT ROW: constants | plot ===
    col_const, _, col_plot = st.columns([1, 1, 3])

    with col_const:
        st.markdown("### Kemeny constants")
        st.metric("Kemeny constant (baseline)", f"{base_k:.3f}")

        if kemeny_defined:
            st.metric(
                "Kemeny constant (after removals)",
                f"{after_k:.3f}",
                delta=f"{delta:+.3f}",
                delta_color="inverse",  # lower is better -> green when delta is negative
            )
        else:
            st.warning("Kemeny constant is undefined for the selected removals.")

    with col_plot:
        st.markdown("### Kemeny constant after each removal")
        fig, ax = plt.subplots()
        series = [base_k] + result.history
        ax.plot(list(range(len(series))), series, marker="o")
        ax.set_xlabel("Step")
        ax.set_ylabel("Kemeny constant")
        ax.grid()
        ax.set_title("Kemeny constant versus removal steps")
        st.pyplot(fig)

    # === BOTTOM: removal order table + reorder controls (1,1,3) ===
    st.markdown("### Removal order")

    if not order:
        st.info("Select edges above to start building a removal order.")
        return

    order_df = pd.DataFrame({"Step": list(range(1, len(order) + 1)), "Edge removed": order})
    order_df = pd.concat(
        [pd.DataFrame({"Step": [0], "Edge removed": ["Baseline (no removal)"]}), order_df],
        ignore_index=True,
    )

    col_a, col_b, col_graph = st.columns([1, 1, 3])

    with col_a:
        st.dataframe(order_df, use_container_width=True, hide_index=True)

    with col_b:
        if "kemeny_edge_active" not in st.session_state or st.session_state["kemeny_edge_active"] not in order:
            st.session_state["kemeny_edge_active"] = order[0]

        st.selectbox("Edge to reorder", options=order, key="kemeny_edge_active")

        b1, b2 = st.columns(2)
        with b1:
            st.button("Up", use_container_width=True, on_click=_move_active, args=(-1,))
        with b2:
            st.button("Down", use_container_width=True, on_click=_move_active, args=(+1,))

        st.button("Remove", use_container_width=True, on_click=_remove_active_edge)

    with col_graph:
        st.markdown("### Network view (after removing edges)")
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
            removed_edges=ordered_edges,
        )


if __name__ == "__main__":
    page()
