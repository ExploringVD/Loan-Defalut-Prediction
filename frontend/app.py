"""Streamlit frontend. It only talks to the FastAPI backend (API_URL in .env).

Start the API first, then from the project folder:
    streamlit run frontend/app.py
"""
import streamlit as st

from api import API_URL, ApiClient, ApiError
from views import dashboard, history, new_application

st.set_page_config(page_title="Loan Default Prediction", page_icon="🏦", layout="wide")

for key, value in {"token": None, "username": None, "role": None}.items():
    st.session_state.setdefault(key, value)


def logout() -> None:
    for key in list(st.session_state):
        del st.session_state[key]


def login_page() -> None:
    st.title("🏦 Loan Default Prediction")
    st.caption("Log in with the username and password you were given (see .env for the local users).")
    with st.form("login"):
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        submitted = st.form_submit_button("Log in", type="primary")
    if submitted:
        try:
            result = ApiClient().login(username.strip(), password)
        except ApiError as e:
            st.error(e.message)
            return
        st.session_state.update(token=result["access_token"], username=username.strip(), role=result["role"])
        st.rerun()


def guarded(view):
    """Run a page; if the token expired, go back to the login page with a message."""
    def run():
        try:
            view()
        except ApiError as e:
            if e.status == 401:
                logout()
                st.session_state["login_message"] = e.message
                st.rerun()
            st.error(e.message)
    return run


if not st.session_state["token"]:
    if st.session_state.get("login_message"):
        st.warning(st.session_state.pop("login_message"))
    login_page()
else:
    with st.sidebar:
        st.markdown(f"Logged in as **{st.session_state['username']}** ({st.session_state['role']})")
        st.button("Log out", on_click=logout)
        st.caption(f"API: {API_URL}")
    pages = [
        st.Page(guarded(new_application.page), title="New Application", icon="📝", url_path="new", default=True),
        st.Page(guarded(history.page), title="History", icon="🗂️", url_path="history"),
    ]
    if st.session_state["role"] == "admin":
        pages.append(st.Page(guarded(dashboard.page), title="Dashboard", icon="📊", url_path="dashboard"))
    st.navigation(pages).run()
