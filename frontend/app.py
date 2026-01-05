# app.py
import base64
from pathlib import Path

import streamlit as st

from views import home, asm_response, eeg_diagnosis, mri_detection, soz_localization  

# ---------------- Paths / logo handling ----------------
LOGO_PATH = Path("assets/logo.jpg")
page_icon = str(LOGO_PATH) if LOGO_PATH.exists() else "🧠"

st.set_page_config(
    page_title="NeuroHeaven",
    page_icon=page_icon,
    layout="wide",
    initial_sidebar_state="collapsed",
)
# ---------------- Hide Streamlit default UI elements ----------------
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

# Encode logo for header
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

# ---------------- GLOBAL CSS ONLY ----------------
CSS = """
<style>
.block-container {
    padding-top: 0rem;
    padding-left: 1rem;
    padding-right: 1rem;
}
header[data-testid="stHeader"] {
    background: transparent;
}

/* NAV ---------------------------------------------------------- */
.nh-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.9rem 0;
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

.nh-center-nav {
    display: flex;
    gap: 2.2rem;
    font-size: 0.98rem;
    font-weight: 500;
    color: #4C5A6B;
    margin-left: 15rem;
    margin-right: auto;
}

.nh-center-nav form {
    margin: 0;
    display: inline;
}

.nh-nav-link {
    background: transparent;
    border: none;
    padding: 0;
    font: inherit;
    color: inherit;
    cursor: pointer;
    position: relative;
    padding-bottom: 0.1rem;
}

.nh-nav-link:hover {
    color: #1B2B3C;
}

.nh-nav-link.active {
    color: #74B0D3 !important;
    font-weight: 700;
}

.nh-right {
    display: flex;
    align-items: center;
}

/* HERO --------------------------------------------------------- */
.nh-hero-wrapper {
    width: 100%;
    padding-top: 2.5rem;
    padding-bottom: 2.5rem;
}

.nh-hero {
    padding: 2.2rem 2.5rem 3.2rem 2.5rem;
    border-radius: 24px;
    background: linear-gradient(
        90deg,
        #F4FBFF 0%,
        #FFFFFF 50%,
        #FFFFFF 100%
    );
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

.nh-benefit-card { text-align: center; }

.nh-benefit-icon {
    width: 96px;
    height: 96px;
    border-radius: 999px;
    background: #F3F7FF;
    margin: 0 auto 1.5rem;
    display: flex;
    align-items: center;
    justify-content: center;
}

.nh-benefit-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #1E3A5F;
    margin-bottom: 0.75rem;
}

.nh-benefit-text {
    font-size: 0.98rem;
    color: #4B5563;
    line-height: 1.7;
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

# ---------------- Routing helpers ----------------
page_param = st.query_params.get("page", ["home"])
current_page = page_param[0] if isinstance(page_param, list) else page_param

def nav_link(label: str, slug: str) -> str:
    active_class = "active" if slug == current_page else ""
    return f"""
    <form method="get">
        <input type="hidden" name="page" value="{slug}">
        <button type="submit" class="nh-nav-link {active_class}">{label}</button>
    </form>
    """

# ---------------- NAVBAR HTML ----------------
st.markdown(f"""
<div class="nh-nav">
<div class="nh-left">
{logo_tag}
<div style="display:flex;align-items:center;">
<span class="nh-brand-text-main">NEURO</span>
<span class="nh-brand-text-sub">HEAVEN</span>
</div>
</div>

<div class="nh-center-nav">

<form method="get" style="display:inline;">
<input type="hidden" name="page" value="home">
<button class="nh-nav-link {'active' if current_page=='home' else ''}">Home</button>
</form>

<form method="get" style="display:inline;">
<input type="hidden" name="page" value="eeg">
<button class="nh-nav-link {'active' if current_page=='eeg' else ''}">EEG Diagnosis</button>
</form>

<form method="get" style="display:inline;">
<input type="hidden" name="page" value="soz">
<button class="nh-nav-link {'active' if current_page=='soz' else ''}">SOZ Localization</button>
</form>

<form method="get" style="display:inline;">
<input type="hidden" name="page" value="mri">
<button class="nh-nav-link {'active' if current_page=='mri' else ''}">MRI Detection</button>
</form>

<form method="get" style="display:inline;">
<input type="hidden" name="page" value="asm">
<button class="nh-nav-link {'active' if current_page=='asm' else ''}">ASM Response</button>
</form>

</div>
</div>
""", unsafe_allow_html=True)


# ---------------- ROUTE TO PAGES ----------------
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
