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

#     # Safe: update widget value in callback
#     st.session_state["kemeny_edge_selected_widget"] = list(new_selected)

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

#     with st.expander("Quick User Guide", expanded=False):
#         col_left_guide, col_right_guide = st.columns([2, 3])

#         with col_left_guide:
#                 st.markdown("### How does this page work?")
#                 st.markdown(
#                     """
#                     This page analyzes the Kemeny constant of the graph, which measures the expected traversal time of a random walk from node a to node b.
#                     When we remove an edge in between two nodes, the Kemeny constant changes.
#                     Depending on the change, we find that the respective connection was deemed as important to the graph.
#                     Specifically:
        
#                     - The (steep) decrease of Kemeny applies that a bottleneck of the system has been removed and the average information traversal time has decreased.
#                     - The (steep) increase of Kemeny applies an important connection has been removed and the average information traversal time has increased.

#                     The recommened workflow is as follows:

#                     1. Observe the Edge sensitivity heatmap on the right. Edges colored red indicate that their removal would increase the Kemeny constant, while blue edges indicate a decrease.
#                     2. Select edges to remove from the multiselect box. The order of removal can be adjusted in the bottom-left panel. The list of highest impact edges can be used as a guide.
#                     3. Observe the Kemeny constant changes and the network structure after removals.
#                     4. Use this information to identify critical connections in the network.
#                     5. Repeat as necessary to explore different removal strategies.



#                     """
#                 )
#         with col_right_guide:
#                 display_network(
#                     G,
#                     node_size=None,
#                     node_color=None,
#                     # highlight=[],
#                     highlight_selected=[],
#                     title="Original graph",
#                     show_labels=True,
#                     removed_edges=None,
#                 )

#     col_left, col_right = st.columns([2, 3])
#     with col_left:
#         st.subheader("Remove edges and observe effect on Kemeny")
#         recompute_on_largest = st.checkbox("Recompute on largest component if disconnected", value=True)

#         label_to_edge = _build_label_to_edge(G)
#         all_labels = list(label_to_edge.keys())

#         # Source of truth for selection
#         if "kemeny_edge_selected_state" not in st.session_state:
#             st.session_state["kemeny_edge_selected_state"] = []
#         if "kemeny_edge_selected_widget" not in st.session_state:
#             st.session_state["kemeny_edge_selected_widget"] = list(st.session_state["kemeny_edge_selected_state"])

#         selected_widget = st.multiselect(
#             "Select edges to remove",
#             options=all_labels,
#             default=st.session_state["kemeny_edge_selected_state"],
#             key="kemeny_edge_selected_widget",
#         )

#         st.session_state["kemeny_edge_selected_state"] = list(selected_widget)
#         selected = list(selected_widget)

#         order = _sync_order(selected, label_to_edge)
#         ordered_edges: List[Edge] = [label_to_edge[lbl] for lbl in st.session_state.get("kemeny_edge_order", [])]

#         base_k = kemeny_constant(G)
#         result = interactive_kemeny_edges(G, ordered_edges, recompute_on_largest)

#         # Kemeny interpretation:
#         # Lower Kemeny means faster mixing / shorter expected travel time in the Markov chain,
#         # so "better" here is LOWER.
#         after_k = result.kemeny
#         kemeny_defined = after_k == after_k  # not NaN
#         delta = (after_k - base_k) if kemeny_defined else None
        
#     with col_right:
#         st.markdown("## Edge impact on Kemeny constant")
#         st.markdown("This graph shows the effect each edge has on the Kemeny constant. " \
#         "Note that the edges are colored by how much the Kemeny constant would change if removed. " \
#         "A red edge indicates a large increase in Kemeny constant and a blue edge indicates a decrease.")

#         G_heat = G.copy()

#         for u, v in ordered_edges:
#             if G_heat.has_edge(u, v):
#                 G_heat.remove_edge(u, v)
#             elif not G_heat.is_directed() and G_heat.has_edge(v, u):
#                 G_heat.remove_edge(v, u)
        
#         current_heat_k = kemeny_constant(G_heat)

#         edge_impacts = {}
#         for e in G_heat.edges():
#             result_heat = interactive_kemeny_edges(G_heat, [e], recompute_on_largest)
#             if result_heat.kemeny == result_heat.kemeny:
#                 edge_impacts[e] = result_heat.kemeny - current_heat_k
#             else:
#                 edge_impacts[e] = None

#         display_network(
#             G_heat,
#             edge_color = edge_impacts,
#             title = "Edge sensitivity heatmap (Change in Kemeny if removed)",
#             show_labels = True,
#         )
        
#     with col_left:
#         st.markdown("### List of highest impact edges")
#         st.markdown("This table shows the edges with the highest impact on the Kemeny constant if removed. " \
#                     "Use this as a guide to select edges to remove from the network.")
#         if not edge_impacts:
#             st.info("No edges available.")
#             return
#         impact_items = [
#             (e, delta_k)
#             for e, delta_k in edge_impacts.items()
#             if delta_k is not None
#         ]
#         impact_items.sort(key=lambda x: abs(x[1]), reverse=True)
#         impact_df = pd.DataFrame(
#             {
#                 "Edge": [_edge_label(G, e) for e, _ in impact_items],
#                 "Change in Kemeny": [delta_k for _, delta_k in impact_items],
#             }
#         )
#         st.dataframe(impact_df, use_container_width=True, hide_index=True)


#     # === MAIN CONTENT ROW: constants | plot ===
#     col_const, col_b, col_plot = st.columns([1, 1, 3])

#     with col_const:
#         st.markdown("### Kemeny constants")
#         st.metric("Kemeny constant (baseline)", f"{base_k:.3f}")

#         if kemeny_defined:
#             if selected_widget:
#                     st.metric(
#                         "Kemeny constant (after removals)",
#                         f"{after_k:.3f}",
#                         delta=f"{delta:+.3f}",
#                         delta_color="inverse",  # lower is better -> green when delta is negative
#                     )
#         else:
#             st.warning("Kemeny constant is undefined for the selected removals.")

#     with col_plot:
#         if order:
#                 st.markdown("### Kemeny constant after each removal")
#                 st.markdown("The order of the removal of certain edges has an impact on the subsequent Kemeny values that the remaining edges contain. A certain edge can have a bigger impact on the information network if similar edges have already been removed. As such, the following graph shows the inpact of each edge removal per step.")
#                 fig, ax = plt.subplots()
#                 series = [base_k] + result.history
#                 ax.plot(list(range(len(series))), series, marker="o")
#                 ax.set_xlabel("Step")
#                 ax.set_ylabel("Kemeny constant")
#                 ax.grid()
#                 ax.set_title("Kemeny constant versus removal steps")
#                 st.pyplot(fig)


#     # === BOTTOM: removal order table + reorder controls (1,1,3) ===
#     #st.markdown("### Removal order")

#     if not order:
#         st.info("Select edges above to start building a removal order.")
#         return

#     order_df = pd.DataFrame({"Step": list(range(1, len(order) + 1)), "Edge removed": order})
#     order_df = pd.concat(
#         [pd.DataFrame({"Step": [0], "Edge removed": ["Baseline (no removal)"]}), order_df],
#         ignore_index=True,
#     )

#     with col_const:
#         st.dataframe(order_df, use_container_width=True, hide_index=True)
    
#     with col_b:
#         if "kemeny_edge_active" not in st.session_state or st.session_state["kemeny_edge_active"] not in order:
#             st.session_state["kemeny_edge_active"] = order[0]

#         st.markdown("<div style='height: 244px;'></div>", unsafe_allow_html=True)
#         st.selectbox("Edge to reorder", options=order, key="kemeny_edge_active")

#         b1, b2 = st.columns(2)
#         with b1:
#             st.button("Up", use_container_width=True, on_click=_move_active, args=(-1,))
#         with b2:
#             st.button("Down", use_container_width=True, on_click=_move_active, args=(+1,))

#         st.button("Remove", use_container_width=True, on_click=_remove_active_edge)

#     col_conclusions, col_graph, _ = st.columns([2, 2, 1])

#     with col_conclusions:
#         st.markdown("### Conclusions")
#         st.markdown(
#             """
#             The Kemeny constant provides insights into the connectivity and information flow within the network.
#             Note that the edge sensitivity heatmap shows a possible clustered structure similar to that observed in the community detection analysis.
#             By selectively removing edges based on their impact on the Kemeny constant, we can identify critical connections that significantly influence the network's overall connectivity.
#             Observing the changes in the Kemeny constant after each removal helps us understand how the network's efficiency is affected, which may inform strategies for enhancing or disrupting information flow within the network.
#             """
#         )
        
#     with col_graph:
#         st.markdown("### Network view (after removing edges)")
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
#             # highlight=[],
#             highlight_selected=[],
#             title="Graph after edge removals",
#             show_labels=True,
#             removed_edges=ordered_edges,
#         )

# if __name__ == "__main__":
#     page()














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

    with st.expander("Quick User Guide", expanded=False):
        col_left_guide, col_right_guide = st.columns([2, 3])

        with col_left_guide:
            st.markdown("### How does this page work?")
            st.markdown(
                """
                    This page analyzes the Kemeny constant of the graph, which measures the expected traversal time of a random walk from node a to node b.
                    When we remove an edge in between two nodes, the Kemeny constant changes.
                    Depending on the change, we find that the respective connection was deemed as important to the graph.
                    Specifically:
        
                    - The (steep) decrease of Kemeny applies that a bottleneck of the system has been removed and the average information traversal time has decreased.
                    - The (steep) increase of Kemeny applies an important connection has been removed and the average information traversal time has increased.

                    The recommened workflow is as follows:

                    1. Observe the Edge sensitivity heatmap on the right. Edges colored red indicate that their removal would increase the Kemeny constant, while blue edges indicate a decrease.
                    2. Select edges to remove from the multiselect box. The order of removal can be adjusted in the bottom-left panel. The list of highest impact edges can be used as a guide.
                    3. Observe the Kemeny constant changes and the network structure after removals.
                    4. Use this information to identify critical connections in the network.
                    5. Repeat as necessary to explore different removal strategies.



                    """
            )
        with col_right_guide:
            display_network(
                G,
                node_size=None,
                node_color=None,
                # highlight=[],
                highlight_selected=[],
                title="Original graph",
                show_labels=True,
                removed_edges=None,
            )

    # === INPUTS + CORE COMPUTATION (kept identical) ===
    recompute_on_largest_default = True

    label_to_edge = _build_label_to_edge(G)
    all_labels = list(label_to_edge.keys())

    # Source of truth for selection
    if "kemeny_edge_selected_state" not in st.session_state:
        st.session_state["kemeny_edge_selected_state"] = []
    if "kemeny_edge_selected_widget" not in st.session_state:
        st.session_state["kemeny_edge_selected_widget"] = list(
            st.session_state["kemeny_edge_selected_state"]
        )

    # === FIRST MAIN SPLIT: left controls + constants | right reorder + plot + network ===
    col_left, col_middle, col_right = st.columns([1, 2, 2])

    with col_left:
        st.subheader("Remove edges and observe effect on Kemeny")
        recompute_on_largest = st.checkbox(
            "Recompute on largest component if disconnected", value=recompute_on_largest_default
        )

        selected_widget = st.multiselect(
            "Select edges to remove",
            options=all_labels,
            default=st.session_state["kemeny_edge_selected_state"],
            key="kemeny_edge_selected_widget",
        )

        st.session_state["kemeny_edge_selected_state"] = list(selected_widget)
        selected = list(selected_widget)

        order = _sync_order(selected, label_to_edge)
        ordered_edges: List[Edge] = [
            label_to_edge[lbl] for lbl in st.session_state.get("kemeny_edge_order", [])
        ]

        base_k = kemeny_constant(G)
        result = interactive_kemeny_edges(G, ordered_edges, recompute_on_largest)

        # Kemeny interpretation:
        # Lower Kemeny means faster mixing / shorter expected travel time in the Markov chain,
        # so "better" here is LOWER.
        after_k = result.kemeny
        kemeny_defined = after_k == after_k  # not NaN
        delta = (after_k - base_k) if kemeny_defined else None

        st.markdown("### Kemeny constants")
        st.metric("Kemeny constant (baseline)", f"{base_k:.3f}")

        if kemeny_defined:
            if selected_widget:
                st.metric(
                    "Kemeny constant (after removals)",
                    f"{after_k:.3f}",
                    delta=f"{delta:+.3f}",
                    delta_color="inverse",  # lower is better -> green when delta is negative
                )
        else:
            st.warning("Kemeny constant is undefined for the selected removals.")

        if not order:
            st.info("Select edges above to start building a removal order.")
            return


        if "kemeny_edge_active" not in st.session_state or st.session_state["kemeny_edge_active"] not in order:
            st.session_state["kemeny_edge_active"] = order[0]

        st.selectbox("Edge to reorder", options=order, key="kemeny_edge_active")

        b1, b2 = st.columns(2)
        with b1:
            st.button("Up", use_container_width=True, on_click=_move_active, args=(-1,))
        with b2:
            st.button("Down", use_container_width=True, on_click=_move_active, args=(+1,))

        st.button("Remove", use_container_width=True, on_click=_remove_active_edge)
            
        order_df = pd.DataFrame({"Step": list(range(1, len(order) + 1)), "Edge removed": order})
        order_df = pd.concat(
            [pd.DataFrame({"Step": [0], "Edge removed": ["Baseline (no removal)"]}), order_df],
            ignore_index=True,
        )
        st.dataframe(order_df, use_container_width=True, hide_index=True)

    with col_middle:
        if order:
            st.markdown("### Kemeny constant after each removal")
            st.markdown(
                "The order of the removal of certain edges has an impact on the subsequent Kemeny values that the remaining edges contain. A certain edge can have a bigger impact on the information network if similar edges have already been removed. As such, the following graph shows the inpact of each edge removal per step."
            )
            fig, ax = plt.subplots()
            series = [base_k] + result.history
            ax.plot(list(range(len(series))), series, marker="o")
            ax.set_xlabel("Step")
            ax.set_ylabel("Kemeny constant")
            ax.grid()
            ax.set_title("Kemeny constant versus removal steps")
            st.pyplot(fig)

        
    with col_right:
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
            # highlight=[],
            highlight_selected=[],
            title="Graph after edge removals",
            show_labels=True,
            removed_edges=ordered_edges,
        )

    # === HEATMAP GRAPH COMPUTATION (kept identical) ===
    G_heat = G.copy()

    for u, v in ordered_edges:
        if G_heat.has_edge(u, v):
            G_heat.remove_edge(u, v)
        elif not G_heat.is_directed() and G_heat.has_edge(v, u):
            G_heat.remove_edge(v, u)

    current_heat_k = kemeny_constant(G_heat)

    edge_impacts = {}
    for e in G_heat.edges():
        result_heat = interactive_kemeny_edges(G_heat, [e], recompute_on_largest)
        if result_heat.kemeny == result_heat.kemeny:
            edge_impacts[e] = result_heat.kemeny - current_heat_k
        else:
            edge_impacts[e] = None

    # === SECOND SPLIT: left highest-impact table | right heatmap ===
    col_left_2, col_right_2 = st.columns([2, 3])

    with col_left_2:
        st.markdown("### List of highest impact edges")
        st.markdown(
            "This table shows the edges with the highest impact on the Kemeny constant if removed. "
            "Use this as a guide to select edges to remove from the network."
        )
        if not edge_impacts:
            st.info("No edges available.")
            return
        impact_items = [(e, delta_k) for e, delta_k in edge_impacts.items() if delta_k is not None]
        impact_items.sort(key=lambda x: abs(x[1]), reverse=True)
        impact_df = pd.DataFrame(
            {
                "Edge": [_edge_label(G, e) for e, _ in impact_items],
                "Change in Kemeny": [delta_k for _, delta_k in impact_items],
            }
        )
        st.dataframe(impact_df, use_container_width=True, hide_index=True)

    with col_right_2:
        st.markdown("## Edge impact on Kemeny constant")
        st.markdown(
            "This graph shows the effect each edge has on the Kemeny constant. "
            "Note that the edges are colored by how much the Kemeny constant would change if removed. "
            "A red edge indicates a large increase in Kemeny constant and a blue edge indicates a decrease."
        )

        display_network(
            G_heat,
            edge_color=edge_impacts,
            title="Edge sensitivity heatmap (Change in Kemeny if removed)",
            show_labels=True,
        )

    st.markdown("### Conclusions")
    st.markdown(
        """
            The Kemeny constant provides insights into the connectivity and information flow within the network.
            Note that the edge sensitivity heatmap shows a possible clustered structure similar to that observed in the community detection analysis.
            By selectively removing edges based on their impact on the Kemeny constant, we can identify critical connections that significantly influence the network's overall connectivity.
            Observing the changes in the Kemeny constant after each removal helps us understand how the network's efficiency is affected, which may inform strategies for enhancing or disrupting information flow within the network.
            """
    )


if __name__ == "__main__":
    page()
