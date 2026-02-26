import base64
import json
from pathlib import Path

import streamlit as st
import extra_streamlit_components as stx

from utils.google_oauth import GoogleOAuth
from utils.api_client import post, get

from views import home, asm_response, eeg_diagnosis, mri_detection, soz_localization, auth_page


LOGO_PATH = Path("assets/logo.jpg")
page_icon = str(LOGO_PATH) if LOGO_PATH.exists() else "🧠"

st.set_page_config(
    page_title="NeuroHeaven",
    page_icon=page_icon,
    layout="wide",
    initial_sidebar_state="collapsed",
)

def handle_google_callback():
    code = st.query_params.get("code")
    if not code:
        return

    google_oauth = GoogleOAuth()

    result = google_oauth.handle_callback(
        code if isinstance(code, str) else code[0]
    )

    if result["success"] and result["user"]:
        user_info = result["user"]

        data = post("/auth/google-login", {
            "email": user_info["email"],
            "name": user_info["name"],
            "picture": user_info["picture"],
            "google_id": user_info["google_id"]
        })

        st.session_state["token"] = data["access_token"]
        st.session_state["user"] = get(
            "/auth/me",
            token=st.session_state["token"]
        )

        # Clear URL params so it doesn’t re-trigger
        st.query_params.clear()

        st.rerun()

    else:
        st.query_params.clear()
        st.error("Google authentication failed.")

handle_google_callback()

# ---------------------------
# COOKIE-BASED AUTH PERSISTENCE
# ---------------------------
if "nh_cm_manager" not in st.session_state:
    st.session_state["nh_cm_manager"] = stx.CookieManager(key="nh_cookie_manager_v5")

cm = st.session_state["nh_cm_manager"]
st.session_state["_nh_cm"] = cm  # keep for auth_page.py compatibility if you use it


def _restore_auth_from_cookies():
    # Don't override existing session auth
    if st.session_state.get("token"):
        return

    try:
        auth_cookie = cm.get("nh_auth")
    except Exception:
        # Component not ready yet
        st.stop()

    if not auth_cookie:
        return

    try:
        auth = json.loads(auth_cookie) if isinstance(auth_cookie, str) else auth_cookie
        token_cookie = auth.get("token")
        user_obj = auth.get("user") or {}

        if token_cookie:
            st.session_state["token"] = token_cookie
            st.session_state["user"] = user_obj
            st.rerun()
    except Exception:
        # corrupted cookie / unexpected shape -> ignore
        pass


_restore_auth_from_cookies()

# --------------------------------------------------
# NAVIGATION STATE & ROUTING
# --------------------------------------------------
if "page" not in st.session_state:
    st.session_state["page"] = st.query_params.get("page", "home")


def go_to(page_key: str):
    st.session_state["page"] = page_key
    st.query_params["page"] = page_key
    st.rerun()


current_page = st.session_state["page"]

# --------------------------------------------------
# AUTH GUARD
# --------------------------------------------------
PUBLIC_PAGES = {"home"}
token = st.session_state.get("token")

if (not token) and (current_page not in PUBLIC_PAGES):
    st.session_state["pending_page"] = current_page
    auth_page.open("signin")
    st.session_state["page"] = "home"
    st.query_params["page"] = "home"
    st.rerun()

# Hide Streamlit default UI elements
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {display: none;}
    [data-testid="stSidebarNav"] {display: none;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Encode logo
if LOGO_PATH.exists():
    logo_b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode("utf-8")
    logo_tag = f'<img class="nh-logo" src="data:image/jpeg;base64,{logo_b64}" />'
else:
    logo_tag = (
        '<div class="nh-logo" '
        'style="border-radius:12px;background:#4A7DFF;'
        'color:white;display:flex;align-items:center;'
        'justify-content:center;font-weight:700;">NH</div>'
    )

# --------------------------------------------------
# CSS (merged: your current CSS + "previous navbar" styles)
# NOTE: We keep your layout/hero css, and ADD the nav-link/auth button look
# for Streamlit buttons so they visually match the old HTML navbar.
# --------------------------------------------------
CSS = """
<style>
.block-container {
    padding-top: 0rem;
    padding-left: 1rem;
    padding-right: 1rem;
}
header[data-testid="stHeader"] { background: transparent; }

/* NAV BAR CONTAINER (same as old) */
.nh-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.9rem 1rem;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;

    background: rgba(255, 255, 255, 0.96);
    border-bottom: 1px solid rgba(226, 232, 240, 0.9);
    box-shadow: 0 10px 25px rgba(15, 52, 96, 0.06);
    position: sticky;
    top: 0;
    z-index: 50;
    backdrop-filter: blur(12px);
}

.nh-left {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    min-width: fit-content;
}

.nh-logo {
    width: 42px;
    height: 42px;
    border-radius: 12px;
    object-fit: cover;
}

.nh-brand-text-main {
    font-weight: 800;
    font-size: 1.6rem;
    letter-spacing: 0.02em;
    color: #1E3A5F;
}
.nh-brand-text-sub {
    font-weight: 800;
    font-size: 1.6rem;
    color: #74B0D3;
    margin-left: 0.15rem;
}

/* Center nav area */
.nh-center-nav {
    display: flex;
    gap: 2.2rem;
    font-size: 0.98rem;
    font-weight: 500;
    color: #4C5A6B;
    align-items: center;
    justify-content: center;
    flex: 1;
}

/* Right auth area */
.nh-right {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: 0.6rem;
    min-width: fit-content;
}
.nh-user {
    font-weight: 600;
    color: #1E3A5F;
    margin-right: 0.25rem;
}

/* -----------------------------------------------------------------
   IMPORTANT: Make Streamlit buttons LOOK like the old HTML navbar
   ----------------------------------------------------------------- */

   
/* Base button reset for navbar buttons */
.nh-nav-btn div.stButton > button,
.auth-container div.stButton > button {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
    min-height: 0 !important;
    height: auto !important;
    font: inherit !important;
    cursor: pointer !important;
    text-transform: none !important;
}

/* NAV links look */
.nh-nav-btn div.stButton > button {
    color: #ffffff !important;
    font-weight: 500 !important;
    font-size: 0.98rem !important;
    white-space: nowrap !important;
    padding-bottom: 0.1rem !important;
}
.nh-nav-btn div.stButton > button:hover { color: #1B2B3C !important; }

/* Active nav (using Streamlit primary type) */
.nh-nav-btn div.stButton > button[kind="primary"] {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* Auth buttons: mimic old .nh-btn primary/secondary pills */
.auth-container div.stButton > button {
    padding: 0.45rem 1rem !important;
    border-radius: 999px !important;
    font-size: 0.95rem !important;
    line-height: 1 !important;
}

/* Signup (secondary) */
.btn-signup div.stButton > button {
    color: #4C5A6B !important;
    font-weight: 600 !important;
    margin-top: -1rem !important;
}

/* Login (primary gradient pill) */
.btn-login div.stButton > button {
    background: linear-gradient(90deg, #00C2FF 0%, #2D71FF 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    box-shadow: 0 8px 22px rgba(45,113,255,0.18) !important;
    margin-top: -1rem !important;
}

/* Logout: match old simple button (not pill) */
.btn-logout div.stButton > button {
    color: #475569 !important;
    font-weight: 600 !important;
    padding: 0.25rem 0.5rem !important;
    border-radius: 8px !important;
}
.btn-logout div.stButton > button:hover { opacity: 0.92 !important; }

/* Remove extra vertical spacing Streamlit adds around buttons */
.nh-nav-btn div.stButton { line-height: 1 !important; }
.auth-container div.stButton { line-height: 1 !important; }

/* Keep nav tight */
div[data-testid="column"] > div:has(.nh-nav-btn) { padding-top: 0.15rem; }

/* Optional: ensure button focus outline doesn't look ugly */
.nh-nav-btn div.stButton > button:focus,
.auth-container div.stButton > button:focus {
    outline: none !important;
}

/* --------------------------------------------------
   (Your existing HERO / sections CSS kept as-is)
--------------------------------------------------- */

/* HERO --------------------------------------------------------- */
.nh-hero-wrapper {
    width: 100%;
    padding-top: 2.5rem;
    padding-bottom: 2.5rem;
}

.nh-hero {
    padding: 2.2rem 2.5rem 3.2rem 2.5rem;
    border-radius: 24px;
    background: linear-gradient(90deg, #F4FBFF 0%, #FFFFFF 50%, #FFFFFF 100%);
    box-shadow: 0 18px 40px rgba(15, 52, 96, 0.06);
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
}

.nh-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    border-radius: 999px;
    background: #F3F7FF;
    padding: 0.45rem 1rem;
    font-size: 0.875rem;
    font-weight: 600;
    color: #74B0D3;
    border: 1px solid #74B0D3;
    margin-bottom: 1.4rem;
}

.nh-pill-icon {
    width: 18px;
    height: 18px;
    border-radius: 999px;
    border: 1px solid #4A7DFF33;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.8rem;
    background: #FFFFFF;
}

.nh-hero-title {
    font-size: 5rem !important;
    line-height: 1.05 !important;
    max-width: 1400px !important;
    font-weight: 800 !important;
    color: #1E3A5F !important;
    margin: 0 !important;
    text-align: center !important;
}

.nh-hero-title span:nth-child(2) {
    display: block !important;
    margin-top: 0.5rem !important;
    color: #74B0D3 !important;
}

.nh-hero-subtext {
    margin-top: -4.5rem !important;
    max-width: 830px;
    font-size: 1.2rem !important;
    color: #49576B;
    line-height: 1.7;
}

/* STATS SECTION ------------------------------------------------ */
.nh-stats {
    max-width: 1200px;
    margin: 2rem auto 0 auto;
    border-radius: 24px;
    display: flex;
    justify-content: space-between;
    gap: 2.5rem;
}

.nh-stat-card {
    flex: 1;
    text-align: center;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.nh-stat-card:not(:last-child) {
    border-right: 1px solid #E5EDF7;
}

.nh-stat-icon {
    width: 72px;
    height: 72px;
    border-radius: 24px;
    background: #F3F7FF;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2rem;
    margin: 0 auto 1.5rem;
    color: #1E3A5F;
}

.nh-stat-value {
    font-size: 2.4rem;
    font-weight: 700;
    color:  #1E3A5F;
    margin-bottom: 0rem;
}

.nh-stat-label-main {
    font-size: 1rem;
    font-weight: 700;
    color: #1E3A5F;
}

/* WHY SECTION -------------------------------------------------- */
.nh-why-section {
    padding: 6rem 0 6rem;
    background: linear-gradient(to bottom, rgba(244, 251, 255, 0.7), #ffffff);
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.nh-why-inner {
    max-width: 1200px;
    margin: 0 auto;
    text-align: center;
}

.nh-why-header {
    margin-bottom: 3.5rem;
}

.nh-why-title {
    font-size: 3.5rem !important;
    font-weight: 700 !important;
    color: #1E3A5F;
    margin-top: -1.5rem !important;
    margin-bottom: 0.75rem;
}

.nh-why-subtitle {
    font-size: 1.15rem !important;
    color: #4B5563;
    max-width: 800px;
    justify-content: center !important;
    margin-left: 10rem !important;
    margin: 0 auto;
}

.nh-why-grid {
    display: grid;
    grid-template-columns: repeat(1, minmax(0, 1fr));
    gap: 2.5rem;
}

@media (min-width: 768px) {
    .nh-why-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (min-width: 1024px) {
    .nh-why-grid { grid-template-columns: repeat(4, minmax(0, 1fr)); }
}

/* smaller screens */
@media (max-width: 900px) {
    .nh-stats {
        flex-direction: column;
        padding: 2rem;
    }
    .nh-stat-card:not(:last-child) {
        border-right: none;
        border-bottom: 1px solid #E5EDF7;
        padding-bottom: 1.5rem;
        margin-bottom: 1.5rem;
    }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# --------------------------------------------------
# NAVBAR (Streamlit button logic, now styled like old HTML navbar)
# --------------------------------------------------
t_logo, t_nav, t_auth = st.columns([2.2, 5.8, 2])

with t_logo:
    st.markdown(
        f"""
        <div class="nh-left" style="padding-top:10px;">
            {logo_tag}
            <div style="display:flex;align-items:center;">
                <span class="nh-brand-text-main">NEURO</span>
                <span class="nh-brand-text-sub">HEAVEN</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with t_nav:
    st.markdown('<div class="nh-nav-btn">', unsafe_allow_html=True)

    nav_cols = st.columns(5)
    pages = [
        ("Home", "home"),
        ("EEG Diagnosis", "eeg"),
        ("SOZ Localization", "soz"),
        ("MRI Detection", "mri"),
        ("ASM Response", "asm"),
    ]
    for i, (label, key) in enumerate(pages):
        with nav_cols[i]:
            if st.button(
                label,
                key=f"nav_{key}",
                type="primary" if current_page == key else "secondary",
                use_container_width=True,
            ):
                go_to(key)

    st.markdown("</div>", unsafe_allow_html=True)

with t_auth:
    st.markdown('<div class="auth-container">', unsafe_allow_html=True)

    user = st.session_state.get("user") or {}
    user_name = user.get("full_name") or user.get("email")
    token = st.session_state.get("token")

    if token and user_name:
        a_col1, a_col2 = st.columns([2, 1])
        a_col1.markdown(
            f"<div style='margin-top:10px; font-weight:600; color:#1E3A5F;'>👤 {user_name}</div>",
            unsafe_allow_html=True,
        )
        with a_col2:
            st.markdown('<div class="btn-logout">', unsafe_allow_html=True)
            if st.button("Log out", key="logout_btn"):
                st.session_state.pop("token", None)
                st.session_state.pop("user", None)
                try:
                    cm.delete("nh_auth")
                except Exception:
                    pass
                go_to("home")
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        a_col1, a_col2 = st.columns(2)
        with a_col1:
            st.markdown('<div class="btn-signup">', unsafe_allow_html=True)
            if st.button("Sign Up", key="signup_click"):
                auth_page.open("signup")
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        with a_col2:
            st.markdown('<div class="btn-login">', unsafe_allow_html=True)
            if st.button("Log In", key="login_click"):
                auth_page.open("signin")
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr style='margin-top:0; margin-bottom:1rem; opacity:0.1;'>", unsafe_allow_html=True)

# --------------------------------------------------
# AUTH DIALOG
# --------------------------------------------------
if st.session_state.get("auth_open"):
    auth_page.render_dialog()

# --------------------------------------------------
# ROUTE RENDERING
# --------------------------------------------------
current_page = st.session_state.get("page", "home")

if current_page == "home":
    home.render()
elif current_page == "asm":
    asm_response.render()
elif current_page == "eeg":
    eeg_diagnosis.render()
elif current_page == "soz":
    soz_localization.render()
elif current_page == "mri":
    mri_detection.render()
else:
    home.render()
