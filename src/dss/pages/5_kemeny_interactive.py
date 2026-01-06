"""Streamlit page: Kemeny constant analysis with interactive node removal."""

import streamlit as st
import plotly.graph_objects as go

from dss.ui.state import init_state, get_state
from dss.ui.components import display_network
from dss.analytics.kemeny import kemeny_constant, interactive_kemeny


def page() -> None:
    st.set_page_config(page_title="Kemeny Analysis", layout="wide")
    st.title("Kemeny Constant and Connectivity Analysis")
    init_state()
    G = get_state("graph")
    if G is None:
        st.info("No graph loaded.  Please upload a `.mtx` file on the Upload page.")
        return
    # Compute baseline Kemeny constant
    base_k = kemeny_constant(G)
    st.metric("Kemeny constant (baseline)", f"{base_k:.3f}")
    # Provide an interactive removal interface
    st.subheader("Remove nodes and observe effect on Kemeny")
    nodes = list(G.nodes())
    selected = st.multiselect("Select nodes to remove", nodes, [])
    recompute_on_largest = st.checkbox("Recompute on largest component if disconnected", value=True)
    # Show the current network with removed nodes highlighted
    st.subheader("Network view (removed nodes highlighted)")
    # Build hover text for each node; include whether the node is removed
    hover_dict = {}
    for node in G.nodes():
        if node in selected:
            hover_dict[node] = f"Node {node}<br><b>Removed</b>"
        else:
            hover_dict[node] = f"Node {node}"
    # Determine marker symbols: use 'x' for removed nodes to make removals obvious
    symbol_map = {node: 'x' for node in selected} if selected else None
    display_network(
        G,
        node_size=None,
        node_color=None,
        highlight=selected if selected else [],
        hover=hover_dict,
        title="Removed nodes are outlined in red" if selected else "Network (no removals yet)",
        label_dict=None,
        symbol_dict=symbol_map,
    )
    # Compute and display the Kemeny constant after removals
    if selected:
        result = interactive_kemeny(G, selected, recompute_on_largest)
        if result.kemeny == result.kemeny:  # not NaN
            st.metric("Kemeny constant after removals", f"{result.kemeny:.3f}")
        else:
            st.warning("Kemeny constant is undefined for the selected removals.")
        # Plot the history of Kemeny values including the baseline at x=0
        st.subheader("Kemeny constant after each removal (including baseline)")
        # Build values: baseline followed by each removal step
        k_values = [base_k] + result.history
        x_vals = list(range(len(k_values)))  # 0,1,2,...
        hover_vals = [f"Removed: none<br>Kemeny: {base_k:.4f}"]
        for idx, node in enumerate(selected):
            kv = result.history[idx]
            hover_vals.append(f"Removed: {selected[idx]}<br>Kemeny: {kv:.4f}")
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(x=x_vals, y=k_values, mode='lines+markers', hovertext=hover_vals, hoverinfo='text'))
        fig_line.update_layout(
            xaxis_title="Number of removed nodes",
            yaxis_title="Kemeny constant",
            showlegend=False,
        )
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("Select nodes from the list above to remove them and recompute the Kemeny constant.")


if __name__ == "__main__":
    page()