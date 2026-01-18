# src/dss/ui/auth.py
"""Authentication helpers for the Streamlit DSS.

Uses Streamlit secrets:
    [auth]
    username => Secret in Streamlit for security
    password => Secret in Streamlit for security

Rules:
- Username is case-insensitive
- Password is case-sensitive

Session state keys (managed via dss.ui.state helpers):
- auth_ok: bool
- auth_user: str | None
"""

from __future__ import annotations

from typing import Optional, Tuple

import streamlit as st

from dss.ui.state import get_state, set_state

AUTH_OK_KEY = "auth_ok"
AUTH_USER_KEY = "auth_user"


def read_auth_config() -> Tuple[Optional[str], Optional[str]]:
    """Read (username, password) from Streamlit secrets."""
    auth = st.secrets.get("auth", {})
    username = auth.get("username")
    password = auth.get("password")
    return username, password


def ensure_auth_state() -> None:
    """Ensure auth-related keys exist in session state."""
    if get_state(AUTH_OK_KEY) is None:
        set_state(AUTH_OK_KEY, False)
    if get_state(AUTH_USER_KEY) is None:
        set_state(AUTH_USER_KEY, None)


def is_logged_in() -> bool:
    """Return True if user is logged in for this session."""
    return bool(get_state(AUTH_OK_KEY))


def logout() -> None:
    """Clear auth state and rerun."""
    set_state(AUTH_OK_KEY, False)
    set_state(AUTH_USER_KEY, None)
    st.rerun()


def check_credentials(username_in: str, password_in: str) -> bool:
    """Validate credentials against Streamlit secrets."""
    expected_user, expected_pass = read_auth_config()

    if not expected_user or not expected_pass:
        st.error("Missing Streamlit secrets: [auth] username and password.")
        return False

    user_ok = username_in.strip().lower() == expected_user.strip().lower()
    pass_ok = password_in == expected_pass
    return user_ok and pass_ok


def render_login(title: str = "Login") -> None:
    """Render login form. On success, sets auth state and reruns."""
    st.title(title)

    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username", value="", autocomplete="username")
        password = st.text_input(
            "Password", value="", type="password", autocomplete="current-password"
        )
        submitted = st.form_submit_button("Sign in")

    if not submitted:
        return

    if check_credentials(username, password):
        set_state(AUTH_OK_KEY, True)
        set_state(AUTH_USER_KEY, username.strip())
        st.rerun()

    st.error("Invalid username or password.")


def get_logged_in_user() -> Optional[str]:
    """Return the logged-in username stored in session state."""
    user = get_state(AUTH_USER_KEY)
    return user if isinstance(user, str) and user.strip() else None


def render_logout_button(label: str = "Log out") -> None:
    """Render a logout button in the current container."""
    if st.button(label):
        logout()


def require_login(title: str = "Login") -> bool:
    """Gate access to pages. Renders login if not logged in.

    Returns:
        True if logged in, otherwise False (and login is rendered).
    """
    ensure_auth_state()
    if is_logged_in():
        return True
    render_login(title=title)
    return False
