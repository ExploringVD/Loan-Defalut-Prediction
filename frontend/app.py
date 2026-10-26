"""Streamlit frontend. It only talks to the FastAPI backend (API_URL in .env).

Start the API first, then from the project folder:
    streamlit run frontend/app.py
"""
import html

import streamlit as st

import ui
from api import API_URL, ApiClient, ApiError
from views import batch, dashboard, history, new_application

st.set_page_config(page_title="Loan Default Prediction", page_icon="🏦", layout="wide")
ui.apply_theme()

for key, value in {"token": None, "username": None, "role": None}.items():
    st.session_state.setdefault(key, value)

# Real project numbers for the login page (reports/final_model.md, reports/business_simulation.md, API benchmark)
HIGHLIGHTS = ["Test ROC-AUC 0.952", "75.8% fewer defaults in simulation", "~0.1 s per decision"]
API_STATUS = {"ok": ("online", "API online"), "degraded": ("degraded", "API online, database unreachable"),
              "offline": ("offline", "API offline")}


def logout() -> None:
    for key in list(st.session_state):
        del st.session_state[key]


@st.cache_data(ttl=15, show_spinner=False)
def api_status() -> str:
    """GET /health at most every 15 seconds (the sidebar is drawn on every click)."""
    return ApiClient().health()


def login_page() -> None:
    hero, card = st.columns([1.15, 1], gap="large", vertical_alignment="center")
    with hero, st.container(key="login_hero"):
        chips = "".join(f'<span class="chip">{text}</span>' for text in HIGHLIGHTS)
        st.markdown(
            f'<div class="hero-mark">🏦</div>'
            f'<div class="hero-title">Loan Default Prediction</div>'
            f'<div class="hero-tagline">Score a loan application in seconds - with the reasons behind every decision.</div>'
            f'<div class="chips">{chips}</div>',
            unsafe_allow_html=True)
    with card, st.container(key="login_card"):
        st.subheader("Welcome back", anchor=False)
        st.caption("Log in with the username and password you were given (see .env for the local users).")
        if st.session_state.get("login_message"):
            st.warning(st.session_state.pop("login_message"))
        with st.form("login", border=False):
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Log in", type="primary", width="stretch")
        if submitted:
            try:
                result = ApiClient().login(username.strip(), password)
            except ApiError as e:
                st.error(e.message)
                return
            st.session_state.update(token=result["access_token"], username=username.strip(), role=result["role"])
            st.rerun()


def sidebar() -> None:
    st.logo(str(ui.LOGO_FILE), size="large", icon_image=str(ui.ICON_FILE))
    with st.sidebar:
        username, role = st.session_state["username"], st.session_state["role"]
        status_class, status_text = API_STATUS[api_status()]
        with st.container(key="sidebar_user"):
            st.markdown(
                f'<div class="user-row"><span class="avatar">{html.escape(username[:1].upper())}</span>'
                f'<span class="user-name">{html.escape(username)}</span>'
                f'<span class="role-badge">{html.escape(role)}</span></div>'
                f'<div class="api-status {status_class}"><span class="dot"></span>{status_text}</div>',
                unsafe_allow_html=True)
            st.button("Log out", on_click=logout, icon=":material/logout:", width="stretch")
        st.caption(f"API: {API_URL}")


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
    login_page()
else:
    sidebar()
    pages = [
        st.Page(guarded(new_application.page), title="New Application", icon=":material/edit_note:", url_path="new",
                default=True),
        st.Page(guarded(batch.page), title="Batch Scoring", icon=":material/upload_file:", url_path="batch"),
        st.Page(guarded(history.page), title="History", icon=":material/history:", url_path="history"),
    ]
    if st.session_state["role"] == "admin":
        pages.append(st.Page(guarded(dashboard.page), title="Dashboard", icon=":material/monitoring:",
                             url_path="dashboard"))
    st.navigation(pages).run()
