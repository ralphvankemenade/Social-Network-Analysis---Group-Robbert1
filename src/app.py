"""Streamlit entry point for the DSS multi‑page dashboard."""

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


def main() -> None:
    st.set_page_config(page_title="DSS Social Network Analysis", layout="wide")
    st.sidebar.title("Navigation")
    pages = {
        "User Manual": _7_user_manual.page,
        "Upload & Overview": _1_upload_overview.page,
        "Centrality Analysis": _2_centrality.page,
        "Role Identification": _3_roles.page,
        "Communities & Robustness": _4_communities_robustness.page,
        "Kemeny Analysis": _5_kemeny_interactive.page,
        "Arrest Optimisation": _6_arrest_optimization.page,
    }
    page_name = st.sidebar.radio("Go to", list(pages.keys()), index=0)
    # Execute the selected page
    pages[page_name]()


if __name__ == "__main__":
    main()
