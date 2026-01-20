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
`st.session_state`. Pages should use these helpers instead of accessing
the session dictionary directly to avoid key errors and to provide
consistent defaults across the application.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import streamlit as st


def init_state() -> None:
    """Initialise session state variables if they are missing."""
    defaults: Dict[str, Any] = {
        # Core data
        "graph": None,
        "adjacency": None,

        # Remember the uploaded .mtx so we can restore without asking the user again
        "mtx_filename": None,   # type: Optional[str]
        "mtx_bytes": None,      # type: Optional[bytes]

        # Analysis caches
        "centrality_table": None,
        "centrality_result": None,
        "role_result": None,
        "community_results": {},  # keyed by method
        "kemeny_result": None,
        "arrest_result": None,

        # Auth
        "auth_ok": False,
        "auth_user": None,
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


def clear_network_state() -> None:
    """Clear the stored network and all derived results."""
    st.session_state["graph"] = None
    st.session_state["adjacency"] = None

    st.session_state["centrality_table"] = None
    st.session_state["centrality_result"] = None
    st.session_state["role_result"] = None
    st.session_state["community_results"] = {}
    st.session_state["kemeny_result"] = None
    st.session_state["arrest_result"] = None


def clear_stored_file() -> None:
    """Clear the stored .mtx file bytes and filename."""
    st.session_state["mtx_filename"] = None
    st.session_state["mtx_bytes"] = None


def reset_all_results() -> None:
    """Reset cached results, but keep the loaded network."""
    st.session_state["centrality_table"] = None
    st.session_state["centrality_result"] = None
    st.session_state["role_result"] = None
    st.session_state["community_results"] = {}
    st.session_state["kemeny_result"] = None
    st.session_state["arrest_result"] = None


if __name__ == "__main__":
    init_state()
    print(dict(st.session_state))
