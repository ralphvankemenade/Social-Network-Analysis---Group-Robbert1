# """Manage Streamlit session state for the DSS.

# This module centralises access to and initialisation of objects stored in
# `st.session_state`.  Pages should use these helpers instead of
# accessing the session dictionary directly to avoid key errors and to
# provide consistent defaults across the application.
# """

# from typing import Any, Dict
# import streamlit as st


# def init_state() -> None:
#     """Initialise session state variables if they are missing."""
#     defaults: Dict[str, Any] = {
#         "graph": None,
#         "adjacency": None,
#         "centrality_table": None,
#         "centrality_result": None,
#         "role_result": None,
#         "community_results": {},
#         "kemeny_result": None,
#         "arrest_result": None,
#         # Auth
#         "auth_ok": False,
#         "auth_user": None,
#     }
#     for key, value in defaults.items():
#         if key not in st.session_state:
#             st.session_state[key] = value


# def get_state(key: str) -> Any:
#     """Helper to get a value from session state."""
#     return st.session_state.get(key)


# def set_state(key: str, value: Any) -> None:
#     """Helper to set a value in session state."""
#     st.session_state[key] = value


# if __name__ == "__main__":
#     # Demonstrate usage in a non‑Streamlit context
#     # (This will not actually persist between runs)
#     init_state()
#     print(st.session_state)





"""Manage Streamlit session state for the DSS.

This module centralises access to and initialisation of objects stored in
`st.session_state`. Pages should use these helpers instead of
accessing the session dictionary directly to avoid key errors and to
provide consistent defaults across the application.
"""

from typing import Any, Dict, Iterable, Optional
import streamlit as st


def init_state() -> None:
    """Initialise session state variables if they are missing."""
    defaults: Dict[str, Any] = {
        "graph": None,
        "adjacency": None,
        "centrality_table": None,
        "centrality_result": None,
        "role_result": None,
        "community_results": {},
        "kemeny_result": None,
        "arrest_result": None,
        # Auth
        "auth_ok": False,
        "auth_user": None,
        # Upload tracking
        "current_file_id": None,
        "last_objective": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_states(keys: Iterable[str]) -> None:
    """Remove a list of keys from session state if they exist."""
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]


def clear_graph_state() -> None:
    """Clear all graph-dependent state.

    This keeps authentication and other global UI state intact.
    """
    keys_to_clear = [
        "graph",
        "adjacency",
        "layout",
        "centrality_table",
        "centrality_result",
        "role_result",
        "community_results",
        "kemeny_result",
        "arrest_result",
        "removed_edges",
        "highlight_top",
        "highlight_selected",
        "highlight_arrested",
        "last_objective"
    ]
    clear_states(keys_to_clear)


def get_state(key: str) -> Any:
    """Helper to get a value from session state."""
    return st.session_state.get(key)


def set_state(key: str, value: Any) -> None:
    """Helper to set a value in session state."""
    st.session_state[key] = value


if __name__ == "__main__":
    # Demonstrate usage in a non-Streamlit context
    # (This will not actually persist between runs)
    init_state()
    print(st.session_state)
