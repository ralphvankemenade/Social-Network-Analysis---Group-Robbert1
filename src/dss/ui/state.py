"""Manage Streamlit session state for the DSS.

This module centralises access to and initialisation of objects stored in
`st.session_state`.  Pages should use these helpers instead of
accessing the session dictionary directly to avoid key errors and to
provide consistent defaults across the application.
"""

from typing import Any, Dict
import streamlit as st


def init_state() -> None:
    """Initialise session state variables if they are missing."""
    defaults: Dict[str, Any] = {
        "graph": None,
        "adjacency": None,
        "centrality_table": None,
        "centrality_result": None,
        "role_result": None,
        "community_results": {},  # keyed by method
        "kemeny_result": None,
        "arrest_result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_state(key: str) -> Any:
    """Helper to get a value from session state."""
    return st.session_state.get(key)


def set_state(key: str, value: Any) -> None:
    """Helper to set a value in session state."""
    st.session_state[key] = value


if __name__ == "__main__":
    # Demonstrate usage in a non‑Streamlit context
    # (This will not actually persist between runs)
    init_state()
    print(st.session_state)