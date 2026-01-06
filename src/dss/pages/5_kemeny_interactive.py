"""Streamlit page: Kemeny constant analysis with interactive node removal."""

import streamlit as st
import matplotlib.pyplot as plt

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
    # Compute and display the Kemeny constant after removals
    if selected:
        result = interactive_kemeny(G, selected, recompute_on_largest)
        if result.kemeny == result.kemeny:  # not NaN
            st.metric("Kemeny constant after removals", f"{result.kemeny:.3f}")
        else:
            st.warning("Kemeny constant is undefined for the selected removals.")
        # Plot the history of Kemeny values as nodes are removed sequentially
        st.subheader("Kemeny constant after each removal")
        fig, ax = plt.subplots()
        # x_vals = list(range(1, len(result.history) + 1))
        # ax.plot(x_vals, result.history, marker="o")
        # Include baseline as starting point
        kemeny_series = [base_k] + result.history
        x_vals = list(range(0, len(kemeny_series)))
        
        ax.plot(x_vals, kemeny_series, marker="o")
        # ax.axvline(0, linestyle="--", linewidth=1)
        ax.set_xlabel("Number of removed nodes")
        ax.set_ylabel("Kemeny constant")
        ax.set_title("Kemeny constant versus number of removed nodes")
        st.pyplot(fig)
        
        # Show the current network with removed nodes highlighted
        st.subheader("Network view (removed nodes highlighted)")
        display_network(
            G,
            node_size=None,
            node_color=None,
            highlight=selected,
            title="Removed nodes are outlined in red",
            show_labels=True,
        )
    else:
        st.info("Select nodes from the list above to remove them and recompute the Kemeny constant.")
    
    
    


if __name__ == "__main__":
    page()
