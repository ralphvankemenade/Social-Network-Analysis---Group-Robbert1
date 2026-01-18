# """Streamlit entry point for the DSS multi‑page dashboard."""

# import streamlit as st

# from dss.pages import (
#     _1_upload_overview,
#     _2_centrality,
#     _3_roles,
#     _4_communities_robustness,
#     _5_kemeny_interactive,
#     _6_arrest_optimization,
#     _7_user_manual,
# )


# def main() -> None:
#     st.set_page_config(page_title="DSS Social Network Analysis", layout="wide")
#     st.sidebar.title("Navigation")
#     pages = {
#         "User Manual": _7_user_manual.page,
#         "Upload & Overview": _1_upload_overview.page,
#         "Centrality Analysis": _2_centrality.page,
#         "Role Identification": _3_roles.page,
#         "Communities & Robustness": _4_communities_robustness.page,
#         "Kemeny Analysis": _5_kemeny_interactive.page,
#         "Arrest Optimisation": _6_arrest_optimization.page,
#     }
#     page_name = st.sidebar.radio("Go to", list(pages.keys()), index=0)
#     # Execute the selected page
#     pages[page_name]()


# if __name__ == "__main__":
#     main()



# src/app.py
"""Streamlit entry point for the DSS multi-page dashboard (with login)."""

from __future__ import annotations

from typing import Callable, Dict

import streamlit as st

from dss.pages import (
    _1_upload_overview,
    _2_centrality,
    _3_roles,
    _4_communities_robustness,
    _5_kemeny_interactive,
    _6_arrest_optimization,
    _7_user_manual,
)
from dss.ui.auth import get_logged_in_user, logout, require_login
from dss.ui.state import init_state


def main() -> None:
    st.set_page_config(page_title="DSS Social Network Analysis", layout="wide")
    init_state()

    if not require_login(title="Login - Social Network Analysis"):
        return

    st.sidebar.title("Navigation")

    user = get_logged_in_user()
    if user:
        st.sidebar.caption(f"Signed in as: {user}")

    if st.sidebar.button("Log out"):
        logout()

    pages: Dict[str, Callable[[], None]] = {
        "User Manual": _7_user_manual.page,
        "Upload & Overview": _1_upload_overview.page,
        "Centrality Analysis": _2_centrality.page,
        "Role Identification": _3_roles.page,
        "Communities & Robustness": _4_communities_robustness.page,
        "Kemeny Analysis": _5_kemeny_interactive.page,
        "Arrest Optimisation": _6_arrest_optimization.page,
    }
    pages[st.sidebar.radio("Go to", list(pages.keys()), index=0)]()


if __name__ == "__main__":
    main()
