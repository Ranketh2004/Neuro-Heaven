import base64
import json
from pathlib import Path

import streamlit as st
import extra_streamlit_components as stx

from utils.google_oauth import GoogleOAuth
from utils.api_client import post, get

from views import home, asm_response, eeg_diagnosis, mri_detection, soz_localization, auth_page, profile


LOGO_PATH = Path("assets/logo.png")
page_icon = str(LOGO_PATH) if LOGO_PATH.exists() else "🧠"

st.set_page_config(
    page_title="NeuroHeaven",
    page_icon=page_icon,
    layout="wide",
    initial_sidebar_state="expanded",
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

        st.query_params.clear()

        st.rerun()

    else:
        st.query_params.clear()
        st.error("Google authentication failed.")

handle_google_callback()

if "nh_cm_manager" not in st.session_state:
    st.session_state["nh_cm_manager"] = stx.CookieManager(key="nh_cookie_manager_v5")

cm = st.session_state["nh_cm_manager"]
st.session_state["_nh_cm"] = cm  

def _restore_auth_from_cookies():
    if st.session_state.get("token"):
        return

    try:
        auth_cookie = cm.get("nh_auth")
    except Exception:
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
        pass

_restore_auth_from_cookies()


def _persist_auth_to_cookie(token: str, user: dict):
    """Persist auth token and user to cookie."""
    if not token:
        return
    try:
        from datetime import datetime, timedelta, timezone
        expires = datetime.now(timezone.utc) + timedelta(days=7)
        auth_payload = json.dumps({"token": token, "user": user})
        cm.set("nh_auth", auth_payload, expires_at=expires)
    except Exception:
        pass


def ls_set(key: str, value: str):
    """No-op placeholder for localStorage set (not available in Streamlit)."""
    pass


def ls_del(key: str):
    """No-op placeholder for localStorage delete (not available in Streamlit)."""
    pass

qp_page = st.query_params.get("page")
if isinstance(qp_page, list):
    qp_page = qp_page[0]

if qp_page:
    st.session_state["page"] = qp_page
elif "page" not in st.session_state:
    st.session_state["page"] = "home"


def go_to(page_key: str):
    st.session_state["page"] = page_key
    st.query_params["page"] = page_key
    st.rerun()


def do_logout():
    st.session_state.pop("token", None)
    st.session_state.pop("user", None)
    try:
        cm.delete("nh_auth")
    except Exception:
        pass
    ls_del("nh_token")
    ls_del("nh_user")
    st.query_params.clear()
    st.query_params["page"] = "home"
    st.rerun()


current_page = st.session_state.get("page", "home")
token = st.session_state.get("token")
user = st.session_state.get("user") or {}
user_name = user.get("full_name") or user.get("name") or user.get("email") or "User"

if token:
    _persist_auth_to_cookie(token, user)
    ls_set("nh_token", token)
    ls_set("nh_user", json.dumps(user or {}))

PROTECTED_PAGES = {"eeg", "soz", "mri", "asm", "profile"}
if (not token) and (current_page in PROTECTED_PAGES):
    st.session_state["pending_page"] = current_page
    auth_page.open("signin")
    go_to("home")

logo_img_src = None
if LOGO_PATH.exists():
    logo_b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode("utf-8")
    logo_img_src = f"data:image/png;base64,{logo_b64}"

def _svg(icon_name: str) -> str:
    if icon_name == "home":
        return '<svg class="nh-ico" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M3 10.5L12 3l9 7.5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M5 9.5V21h14V9.5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>'
    if icon_name == "pulse":
        return '<svg class="nh-ico" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M3 12h4l2-6 4 12 2-6h6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>'
    if icon_name == "pill":
        return '<svg class="nh-ico" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M10 14l4-4" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><path d="M7 17a5 5 0 0 1 0-7l3-3a5 5 0 0 1 7 7l-3 3a5 5 0 0 1-7 0Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>'
    if icon_name == "scan":
        return '<svg class="nh-ico" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M7 3H5a2 2 0 0 0-2 2v2" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><path d="M17 3h2a2 2 0 0 1 2 2v2" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><path d="M7 21H5a2 2 0 0 1-2-2v-2" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><path d="M17 21h2a2 2 0 0 0 2-2v-2" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>'
    return '<svg class="nh-ico" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M9 8a3 3 0 0 1 3-3c1.7 0 3 1.3 3 3v.3a2.7 2.7 0 0 1 2 2.6v3.6A3.5 3.5 0 0 1 15.5 20H14" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M9 8v.3A2.7 2.7 0 0 0 7 10.9v3.6A3.5 3.5 0 0 0 10.5 20H12" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M12 5v15" stroke="currentColor" stroke-width="2" stroke-linecap="round" opacity="0.6"/></svg>'


CSS = f"""
<style>
MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header[data-testid="stHeader"] {{ display: none !important; }}
.block-container {{
    padding-top: 0rem !important;
}}

section[data-testid="stSidebar"] [data-testid="stSidebarNav"] {{ display:none !important; }}
{"" if token else "section[data-testid='stSidebar']{display:none !important;} [data-testid='collapsedControl']{display:none !important;}"}

/* Hide the collapse (<<) button inside the sidebar when logged in */
{"[data-testid='stSidebarCollapseButton'], section[data-testid='stSidebar'] button[kind='header']{ display:none !important; }" if token else ""}

/* Force sidebar to always stay visible when logged in (overrides Chrome cached collapsed state) */
{"section[data-testid='stSidebar']{ transform: none !important; min-width: 244px !important; visibility: visible !important; } [data-testid='collapsedControl']{ display:none !important; }" if token else ""}

section[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, #20A0D8 0%, #0E5C7A 100%) !important;
  border-right: 1px solid rgba(255,255,255,0.08) !important;
  box-shadow: 4px 0 12px rgba(0,0,0,0.1) !important;
}}
section[data-testid="stSidebar"] > div {{
  padding-top: 0 !important;
  overflow: visible !important;
}}
section[data-testid="stSidebar"] div[data-testid="stSidebarContent"] {{
  height: 100vh !important;
  display: flex !important;
  flex-direction: column !important;
  overflow: hidden !important;
  padding: 1rem 1.25rem !important;
  box-sizing: border-box !important;
  font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}

.sb-brand {{
  display: flex;
  align-items: center;
  justify-content: flex-start;
  gap: 0.75rem;
  padding: 0 0 1rem 0;
  margin-bottom: 1.25rem;
  border-bottom: 1px solid rgba(255,255,255,0.14);
  overflow: visible !important;
  flex-shrink: 0;
}}
.sb-logo {{
  width: 42px;
  height: 42px;
  border-radius: 10px;
  object-fit: cover;
  flex-shrink: 0;
  box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}}
.sb-brand-name {{
  color: #ffffff;
  font-weight: 700;
  letter-spacing: -0.02em;
  font-size: 1.2rem;
  white-space: nowrap;
  overflow: visible !important;
}}

.nav-shell {{
  flex: 1 1 auto;
  overflow-y: auto;
  padding-right: 0.25rem;
  min-height: 0;
}}
.nav-shell::-webkit-scrollbar {{ width: 6px; }}
.nav-shell::-webkit-scrollbar-thumb {{
  background: rgba(255,255,255,0.18);
  border-radius: 6px;
}}
.nav-shell::-webkit-scrollbar-track {{
  background: rgba(255,255,255,0.05);
}}

.nav-item {{
  position: relative;
  margin-bottom: 0.375rem;
}}
.nh-nav-card {{
  display: flex;
  align-items: center;
  gap: 0.875rem;
  padding: 0.875rem 1.125rem;
  border-radius: 12px;
  border: none;
  background: transparent;
  user-select: none;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  position: relative;
}}
/* Hover: each nav item is in its own st.container() → stVerticalBlock.
   :has(> stButton:hover) fires when the transparent overlay button is hovered. */
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]:has(> [data-testid="stButton"]:hover) .nh-nav-card {{
  background: rgba(255,255,255,0.16) !important;
}}

/* Active state */
.nh-nav-card.active {{
  background: rgba(185, 226, 255, 0.92) !important;
  box-shadow: 0 4px 12px rgba(185, 226, 255, 0.3);
}}

.nh-nav-card.active::before {{
  content: '';
  position: absolute;
  left: -1.25rem;
  top: 50%;
  transform: translateY(-50%);
  width: 4px;
  height: 60%;
  background: #ffffff;
  border-radius: 0 4px 4px 0;
}}

/* Press/click state */
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]:has(> [data-testid="stButton"]:active) .nh-nav-card {{
  transform: scale(0.98);
}}

.nh-ico {{
  width: 20px;
  height: 20px;
  flex: 0 0 auto;
  color: rgba(235, 248, 255, 0.85);
  transition: color 0.3s ease;
}}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]:has(> [data-testid="stButton"]:hover) .nh-ico {{
  color: rgba(255, 255, 255, 0.95);
}}
.nh-nav-card.active .nh-ico {{
  color: rgba(11,42,87,0.95);
}}
.nh-nav-text {{
  display: flex;
  flex-direction: column;
  gap: 0.125rem;
}}
.nh-nav-title {{
  font-weight: 600;
  font-size: 0.9rem;
  color: rgba(255,255,255,0.95);
  line-height: 1.2;
  transition: color 0.3s ease;
}}
.nh-nav-sub {{
  font-weight: 400;
  font-size: 0.75rem;
  color: rgba(255,255,255,0.7);
  line-height: 1.2;
  transition: color 0.3s ease;
}}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]:has(> [data-testid="stButton"]:hover) .nh-nav-title {{
  color: rgba(255,255,255,0.98);
}}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]:has(> [data-testid="stButton"]:hover) .nh-nav-sub {{
  color: rgba(255,255,255,0.85);
}}
.nh-nav-card.active .nh-nav-title {{
  color: #0B2A57;
  font-weight: 700;
}}
.nh-nav-card.active .nh-nav-sub {{
  color: rgba(11,42,87,0.75);
}}

/* User section */
.sb-user {{
  margin-top: 1rem;
  padding-top: 1rem;
  border-top: 1px solid rgba(255,255,255,0.14);
  flex-shrink: 0;
}}
.sb-user-row {{
  color: #ffffff;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 0.75rem;
}}

/* ALL SIDEBAR NAV OVERLAY BUTTONS */
section[data-testid="stSidebar"] [data-testid="stButton"] {{
  position: relative !important;
  margin-top: -60px !important;
  margin-bottom: -10px !important;
  height: 55px !important;
  z-index: 100 !important;
  pointer-events: none !important;
}}
section[data-testid="stSidebar"] [data-testid="stButton"] button {{
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  color: transparent !important;
  height: 55px !important;
  min-height: 0 !important;
  padding: 0 !important;
  pointer-events: auto !important;
}}

/* Reduce Streamlit element gaps in sidebar */
section[data-testid="stSidebar"] [data-testid="stMarkdown"] {{
  margin-bottom: 0 !important;
}}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
  gap: 0.65rem !important;
}}

/* User navigation separator */
.sb-user-nav {{
  margin-top: 1rem;
  padding-top: 1rem;
  border-top: 1px solid rgba(255,255,255,0.14);
  flex-shrink: 0;
}}

/* Profile toggle chevron */
.nh-chevron {{
  width: 16px;
  height: 16px;
  margin-left: auto;
  color: rgba(255,255,255,0.7);
  flex-shrink: 0;
  transition: all 0.3s ease;
}}
.nh-chevron.open {{
  transform: rotate(180deg);
  color: rgba(255,255,255,0.9);
}}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]:has(> [data-testid="stButton"]:hover) .nh-chevron {{
  color: rgba(255,255,255,0.9);
}}

/* Dropdown sub-items */
.sb-dropdown-item {{
  margin-left: 1rem !important;
  margin-bottom: 0.30rem !important;
}}
.sb-dropdown-item .nh-nav-card {{
  background: rgba(255,255,255,0.08) !important;
  padding: 0.625rem 0.875rem;
  border-radius: 8px;
  transition: all 0.3s ease;
}}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]:has(> [data-testid="stButton"]:hover) .sb-dropdown-item .nh-nav-card {{
  background: rgba(255,255,255,0.14) !important;
}}
.sb-dropdown-item .nh-nav-title {{
  font-size: 0.825rem;
  color: rgba(255,255,255,0.85);
}}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]:has(> [data-testid="stButton"]:hover) .sb-dropdown-item .nh-nav-title {{
  color: rgba(255,255,255,0.95);
}}
/* Logout item styling */
.sb-logout-item .nh-nav-card {{
  border-left: 3px solid rgba(239, 68, 68, 0.8);
}}
.sb-logout-item .nh-nav-title {{
  color: rgba(255,160,160,0.95) !important;
}}
.sb-logout-item .nh-ico {{
  color: rgba(255,160,160,0.90) !important;
}}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]:has(> [data-testid="stButton"]:hover) .sb-logout-item .nh-nav-card {{
  background: rgba(255,100,100,0.12) !important;
  border-color: rgba(255,100,100,0.30) !important;
}}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]:has(> [data-testid="stButton"]:hover) .sb-logout-item .nh-nav-title {{
  color: rgba(255, 180, 180, 0.95);
}}

/* HERO --------------------------------------------------------- */
.nh-hero-wrapper {{
    width: 100%;
    padding-top: 0;
    padding-bottom: 1.5rem;
}}

.nh-hero {{
    padding: 1.5rem 2.5rem 2.5rem 2.5rem;
    border-radius: 24px;
    background: linear-gradient(90deg, #F4FBFF 0%, #FFFFFF 50%, #FFFFFF 100%);
    box-shadow: 0 18px 40px rgba(15, 52, 96, 0.06);
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
}}

.nh-pill {{
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
}}

.nh-pill-icon {{
    width: 18px;
    height: 18px;
    border-radius: 999px;
    border: 1px solid #4A7DFF33;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.8rem;
    background: #FFFFFF;
}}

.nh-hero-title {{
    font-size: 5rem !important;
    line-height: 1.05 !important;
    max-width: 1400px !important;
    font-weight: 800 !important;
    color: #1E3A5F !important;
    margin: 0 !important;
    text-align: center !important;
}}

.nh-hero-title span:nth-child(2) {{
    display: block !important;
    margin-top: 0.5rem !important;
    color: #74B0D3 !important;
}}

.nh-hero-subtext {{
    margin-top: -4.5rem !important;
    max-width: 830px;
    font-size: 1.2rem !important;
    color: #49576B;
    line-height: 1.7;
}}

/* STATS SECTION ------------------------------------------------ */
.nh-stats {{
    max-width: 1200px;
    margin: 2rem auto 0 auto;
    border-radius: 24px;
    display: flex;
    justify-content: space-between;
    gap: 2.5rem;
}}

.nh-stat-card {{
    flex: 1;
    text-align: center;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}

.nh-stat-card:not(:last-child) {{
    border-right: 1px solid #E5EDF7;
}}

.nh-stat-icon {{
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
}}

.nh-stat-value {{
    font-size: 2.4rem;
    font-weight: 700;
    color:  #1E3A5F;
    margin-bottom: 0rem;
}}

.nh-stat-label-main {{
    font-size: 1rem;
    font-weight: 700;
    color: #1E3A5F;
}}

/* WHY SECTION -------------------------------------------------- */
.nh-why-section {{
    padding: 6rem 0 6rem;
    background: linear-gradient(to bottom, rgba(244, 251, 255, 0.7), #ffffff);
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}

.nh-why-inner {{
    max-width: 1200px;
    margin: 0 auto;
    text-align: center;
}}

.nh-why-header {{
    margin-bottom: 3.5rem;
}}

.nh-why-title {{
    font-size: 3.5rem !important;
    font-weight: 700 !important;
    color: #1E3A5F;
    margin-top: -1.5rem !important;
    margin-bottom: 0.75rem;
}}

.nh-why-subtitle {{
    font-size: 1.15rem !important;
    color: #4B5563;
    max-width: 800px;
    justify-content: center !important;
    margin-left: 10rem !important;
    margin: 0 auto;
}}

.nh-why-grid {{
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 2rem;
}}

/* Horizontal Navbar for logged-out users */
.nh-navbar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0rem 0rem;
    background: #fff;
    border-bottom: 4px solid #20A0D8;
    margin: 0 -1rem 4rem -1rem; !important;
    width: calc(100% + 2rem);
    box-shadow: 0 6px 24px rgba(14, 92, 122, 0.12);
}}
.nh-navbar-left {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}
.nh-navbar-logo {{
    width: 42px;
    height: 42px;
    border-radius: 10px;
    object-fit: cover;
}}
.nh-navbar-brand {{
    display: flex;
    align-items: center;
}}
.nh-brand-text-main {{
    font-weight: 900;
    font-size: 1.35rem;
    color: #1E3A5F;
    letter-spacing: 0.02em;
}}
.nh-brand-text-sub {{
    font-weight: 700;
    font-size: 1.35rem;
    color: #74B0D3;
    letter-spacing: 0.02em;
}}
.nh-navbar-right {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
}}
/* Navbar auth buttons - pill style with gradient */
[data-testid="stAppViewContainer"] [data-testid="stHorizontalBlock"] [data-testid="stButton"] button {{
    padding: 0.5rem 2rem !important;
    border-radius: 50px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    min-width: 110px !important;
    white-space: nowrap !important;
    transition: all 0.15s ease !important;
    background: linear-gradient(90deg, #00c6fb 0%, #005bea 100%) !important;
    color: #fff !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(0, 91, 234, 0.3) !important;
}}
[data-testid="stAppViewContainer"] [data-testid="stHorizontalBlock"] [data-testid="stButton"] button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(0, 91, 234, 0.4) !important;
}}

/* BENEFIT CARDS ------------------------------------------------ */
.nh-benefit-card {{
    text-align: center;
    padding: 1.5rem;
    display: flex;
    flex-direction: column;
    align-items: center;
}}

.nh-benefit-icon {{
    margin-bottom: 1rem;
    height: 50px;
    display: flex;
    align-items: center;
    justify-content: center;
}}

.nh-benefit-title {{
    font-size: 1.25rem;
    font-weight: 700;
    color: #1E3A5F;
    margin-bottom: 0.5rem;
    min-height: 3rem;
    display: flex;
    align-items: flex-end;
    justify-content: center;
    white-space: nowrap;
}}

.nh-benefit-text {{
    font-size: 0.85rem;
    color: #4B5563;
    line-height: 1.5;
    max-width: 220px;
    margin: 0 auto;
}}

</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# SIDEBAR NAV
if token:
    with st.sidebar:
        brand_logo_html = (
            f'<img class="sb-logo" src="{logo_img_src}" />'
            if logo_img_src
            else '<div class="sb-logo" style="background:rgba(255,255,255,0.12);display:flex;align-items:center;justify-content:center;color:#fff;font-weight:900;">NH</div>'
        )
        st.markdown(
            f'<div class="sb-brand">{brand_logo_html}<div class="sb-brand-name">NEUROHEAVEN</div></div>',
            unsafe_allow_html=True,
        )

        nav_items = [
            ("home", "home", "Home", "Overview"),
            ("eeg", "brain", "EEG Diagnosis", "Epilepsy Diagnosis"),
            ("soz", "pulse", "SOZ Localization", "EEG Graph Analysis"),
            ("mri", "scan", "MRI Detection", "Lesion Analysis"),
            ("asm", "pill", "ASM Predictor", "Treatment Response"),
        ]

        st.markdown('<div class="nav-shell">', unsafe_allow_html=True)

        for key, ico, title, subtitle in nav_items:
            active = "active" if current_page == key else ""
            with st.container():
                st.markdown(
                    f"""
                    <div class="nh-nav-card {active}">
                        {_svg(ico)}
                        <div class="nh-nav-text">
                            <div class="nh-nav-title">{title}</div>
                            <div class="nh-nav-sub">{subtitle}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(" ", key=f"nav_{key}", use_container_width=True):
                    go_to(key)

        is_open = st.session_state.get("sb_profile_open", False)
        chevron_class = "open" if is_open else ""
        active_profile = "active" if current_page == "profile" and not is_open else ""
        with st.container():
            st.markdown(
                f'''<div class="sb-user-nav"><div class="nh-nav-card {active_profile}">
  <svg class="nh-ico" viewBox="0 0 24 24" fill="none" aria-hidden="true">
    <circle cx="12" cy="8" r="4" stroke="currentColor" stroke-width="2"/>
    <path d="M4 20c0-4 3.6-7 8-7s8 3 8 7" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
  </svg>
  <div class="nh-nav-text">
    <div class="nh-nav-title">{user_name}</div>
    <div class="nh-nav-sub">Account</div>
  </div>
  <svg class="nh-chevron {chevron_class}" viewBox="0 0 24 24" fill="none" aria-hidden="true">
    <path d="M6 9l6 6 6-6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
  </svg>
</div></div>''',
                unsafe_allow_html=True,
            )
            if st.button(" ", key="nav_profile_toggle", use_container_width=True):
                st.session_state["sb_profile_open"] = not is_open
                st.rerun()

        if is_open:
            active_pf = "active" if current_page == "profile" else ""
            with st.container():
                st.markdown(
                    f'''<div class="sb-dropdown-item"><div class="nh-nav-card {active_pf}">
  <svg class="nh-ico" viewBox="0 0 24 24" fill="none" aria-hidden="true">
    <circle cx="12" cy="8" r="4" stroke="currentColor" stroke-width="2"/>
    <path d="M4 20c0-4 3.6-7 8-7s8 3 8 7" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
  </svg>
  <div class="nh-nav-text">
    <div class="nh-nav-title">View Profile</div>
  </div>
</div></div>''',
                    unsafe_allow_html=True,
                )
                if st.button(" ", key="nav_profile_goto", use_container_width=True):
                    st.session_state["sb_profile_open"] = False
                    go_to("profile")

            with st.container():
                st.markdown(
                    '''<div class="sb-dropdown-item sb-logout-item"><div class="nh-nav-card">
  <svg class="nh-ico" viewBox="0 0 24 24" fill="none" aria-hidden="true">
    <path d="M9 21H5a2 2 0 01-2-2V5a2 2 0 012-2h4" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    <polyline points="16,17 21,12 16,7" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    <line x1="21" y1="12" x2="9" y2="12" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
  </svg>
  <div class="nh-nav-text">
    <div class="nh-nav-title">Log Out</div>
  </div>
</div></div>''',
                    unsafe_allow_html=True,
                )
                if st.button(" ", key="nav_logout", use_container_width=True):
                    do_logout()

        st.markdown("</div>", unsafe_allow_html=True)  # close nav-shell

if st.session_state.get("auth_open"):
    auth_page.render_dialog()
    if st.session_state.get("token"):
        _persist_auth_to_cookie(st.session_state["token"], st.session_state.get("user") or {})
        ls_set("nh_token", st.session_state["token"])
        ls_set("nh_user", json.dumps(st.session_state.get("user") or {}))

if not token:
    logo_tag = f'<img class="nh-navbar-logo" src="{logo_img_src}" />' if logo_img_src else ''
    
    t_logo, t_spacer, t_auth = st.columns([2.5, 5, 2.5])
    
    with t_logo:
        st.markdown(
            f'''
            <div class="nh-navbar-left">
                {logo_tag}
                <div class="nh-navbar-brand">
                    <span class="nh-brand-text-main">NEURO</span>
                    <span class="nh-brand-text-sub">HEAVEN</span>
                </div>
            </div>
            ''',
            unsafe_allow_html=True,
        )
    
    with t_auth:
        auth_col1, auth_col2 = st.columns(2)
        with auth_col1:
            if st.button("Login", key="home_login_btn", use_container_width=True):
                auth_page.open("signin")
                st.rerun()
        with auth_col2:
            if st.button("Sign Up", key="home_signup_btn", use_container_width=True):
                auth_page.open("signup")
                st.rerun()

current_page = st.session_state.get("page", "home")

if current_page == "home":
    home.render()
elif current_page == "eeg":
    eeg_diagnosis.render()
elif current_page == "soz":
    soz_localization.render()
elif current_page == "mri":
    mri_detection.render()
elif current_page == "asm":
    asm_response.render()
elif current_page == "profile":
    profile.render()
else:
    home.render()