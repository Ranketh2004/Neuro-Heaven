import streamlit as st


DASHBOARD_CSS = """
<style>
/* ── WRAPPER ─────────────────────────────────────────────────────── */
.dash-wrapper {
    padding: 1.75rem 0.5rem 0.75rem 0.5rem;
    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* Tighten Streamlit's default gap between the title block and the columns */
[data-testid="stVerticalBlock"]:has(.dash-header) {
    gap: 0.5rem !important;
}

/* ── HEADER ─────────────────────────────────────────────────────── */
.dash-header {
    margin-bottom: 0;
}

.dash-overline {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: #20A0D8;
    margin-bottom: 0.6rem;
}

.dash-overline-bar {
    display: inline-block;
    width: 22px;
    height: 3px;
    border-radius: 2px;
    background: linear-gradient(90deg, #20A0D8, #0E5C7A);
    flex-shrink: 0;
}

.dash-title {
    font-size: 1.7rem;
    font-weight: 800;
    color: #1E3A5F;
    margin: 0;
    line-height: 1.25;
    letter-spacing: -0.025em;
}

.dash-title-em {
    background: linear-gradient(90deg, #20A0D8, #0E5C7A);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.dash-rule {
    margin-top: 0.9rem;
    height: 2px;
    width: 44px;
    border-radius: 2px;
    background: linear-gradient(90deg, #20A0D8, #0E5C7A);
}

/* ── CARD ────────────────────────────────────────────────────────── */
.dash-card {
    background: #ffffff;
    border-radius: 20px 20px 0 0;
    padding: 1.6rem 1.6rem 1.35rem 1.6rem;
    box-shadow: 0 4px 24px rgba(15, 52, 96, 0.07);
    border: 1px solid #E8F0F8;
    border-bottom: none;
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
    position: relative;
    overflow: hidden;
    margin-bottom: 0;
}

.dash-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    border-radius: 20px 20px 0 0;
}

.dash-card-eeg::before,
.dash-card-soz::before,
.dash-card-mri::before,
.dash-card-asm::before { background: linear-gradient(90deg, #20A0D8, #0E5C7A); }

/* ── ICON ────────────────────────────────────────────────────────── */
.dash-card-icon {
    width: 48px;
    height: 48px;
    border-radius: 13px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

.dash-icon-eeg { background: linear-gradient(135deg, #DBEAFE, #93C5FD); }
.dash-icon-soz { background: linear-gradient(135deg, #DBEAFE, #93C5FD); }
.dash-icon-mri { background: linear-gradient(135deg, #DBEAFE, #93C5FD); }
.dash-icon-asm { background: linear-gradient(135deg, #DBEAFE, #93C5FD); }

/* ── BODY ────────────────────────────────────────────────────────── */
.dash-card-body { flex: 1; }

.dash-card-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1E3A5F;
    margin: 0 0 0.3rem 0;
}

.dash-card-desc {
    font-size: 0.83rem;
    color: #49576B;
    line-height: 1.6;
    margin: 0 0 0.85rem 0;
}

.dash-features {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
    margin-bottom: 0.5rem;
}

.dash-badge {
    display: inline-flex;
    align-items: center;
    padding: 0.2rem 0.55rem;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 600;
    line-height: 1;
}

.dash-badge-eeg { background: #DBEAFE; color: #1D4ED8; }
.dash-badge-soz { background: #DBEAFE; color: #1D4ED8; }
.dash-badge-mri { background: #DBEAFE; color: #1D4ED8; }
.dash-badge-asm { background: #DBEAFE; color: #1D4ED8; }

.dash-input-label {
    font-size: 0.73rem;
    color: #6B7280;
    margin-top: 0.55rem;
    font-weight: 500;
}

.dash-input-value {
    font-size: 0.78rem;
    color: #374151;
    font-weight: 600;
}

/* ── FOOTER ──────────────────────────────────────────────────────── */
.dash-card-footer {
    border-top: 1px solid #F0F4F8;
    padding-top: 0.9rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.dash-model-info {
    font-size: 0.73rem;
    color: #9CA3AF;
    font-weight: 500;
}

.dash-status-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #1E3A5F;
    margin-right: 0.4rem;
    vertical-align: middle;
}

/* ── OPEN BUTTON (Streamlit button below each card) ──────────────── */
[data-testid="stVerticalBlock"]:has(.dash-card-eeg) [data-testid="stButton"] button,
[data-testid="stVerticalBlock"]:has(.dash-card-soz) [data-testid="stButton"] button,
[data-testid="stVerticalBlock"]:has(.dash-card-mri) [data-testid="stButton"] button,
[data-testid="stVerticalBlock"]:has(.dash-card-asm) [data-testid="stButton"] button {
    background: linear-gradient(90deg, #20A0D8, #0E5C7A) !important;
    border-radius: 0 0 16px 16px !important;
    border: none !important;
    box-shadow: 0 6px 20px rgba(32,160,216,0.25) !important;
    color: #fff !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.65rem 1.25rem !important;
    transition: opacity 0.15s ease !important;
    margin-top: -8px !important;
}

[data-testid="stVerticalBlock"]:has(.dash-card) [data-testid="stButton"] button:hover {
    opacity: 0.88 !important;
    transform: none !important;
    box-shadow: inherit !important;
}

/* Increase vertical gap between card rows */
[data-testid="column"] > div:has(.dash-card) {
    margin-bottom: 3rem !important;
}
</style>
"""


def _eeg_icon():
    return '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#1D4ED8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12h4l2-6 4 12 2-6h6"/></svg>'


def _soz_icon():
    return '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#1D4ED8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M4.22 4.22l2.12 2.12M17.66 17.66l2.12 2.12M2 12h3M19 12h3M4.22 19.78l2.12-2.12M17.66 6.34l2.12-2.12"/></svg>'


def _mri_icon():
    return '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#1D4ED8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="12" cy="12" r="4"/><path d="M3 9h2M3 15h2M19 9h2M19 15h2M9 3v2M15 3v2M9 19v2M15 19v2"/></svg>'


def _asm_icon():
    return '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#1D4ED8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 14l4-4"/><path d="M7 17a5 5 0 0 1 0-7l3-3a5 5 0 0 1 7 7l-3 3a5 5 0 0 1-7 0Z"/></svg>'


CARDS = [
    {
        "key": "eeg",
        "color": "eeg",
        "icon_fn": _eeg_icon,
        "title": "EEG Seizure Diagnosis",
        "desc": "Upload EDF recordings to detect and quantify seizure activity.",
        "badges": ["EDF Upload", "Temporal Timeline", "Probability Score"],
        "input": ".edf files (up to 30 MB)",
        "label": "Open EEG Diagnosis →",
    },
    {
        "key": "soz",
        "color": "soz",
        "icon_fn": _soz_icon,
        "title": "SOZ Localization",
        "desc": "Identify the Seizure Onset Zones.",
        "badges": ["EDF Upload", "Channel Mapping", "Topomap"],
        "input": ".edf (up to 500 MB)",
        "label": "Open SOZ Localization →",
    },
    {
        "key": "mri",
        "color": "mri",
        "icon_fn": _mri_icon,
        "title": "MRI FCD Detection",
        "desc": "Detect Focal Cortical Dysplasia in MRI.",
        "badges": ["NIfTI", "Grad-CAM", "FCD Probability"],
        "input": ".nii, .nii.gz",
        "label": "Open MRI Detection →",
    },
    {
        "key": "asm",
        "color": "asm",
        "icon_fn": _asm_icon,
        "title": "ASM Response Predictor",
        "desc": "Predict seizure freedom probability and rank Anti-Seizure Medication.",
        "badges": ["17 Clinical Fields", "ASM Ranking", "Seizure Freedom %"],
        "input": "Patient demographics & clinical history",
        "label": "Open ASM Predictor →",
    },
]


def render():
    st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="dash-wrapper">
          <div class="dash-header">
            <h2 class="dash-title">
              Explore specialized <span class="dash-title-em">AI modules</span>
            </h2>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


    for row_start in range(0, len(CARDS), 2):
        row_cards = CARDS[row_start:row_start + 2]
        cols = st.columns(2, gap="medium")

        for col, card in zip(cols, row_cards):
            with col:
                color = card["color"]
                badges_html = "".join(
                    f'<span class="dash-badge dash-badge-{color}">{b}</span>'
                    for b in card["badges"]
                )

                st.markdown(
                    f"""
                    <div class="dash-card dash-card-{color}">
                      <div style="display:flex;align-items:flex-start;gap:0.9rem;">
                        <div class="dash-card-icon dash-icon-{color}">{card["icon_fn"]()}</div>
                        <div class="dash-card-body">
                          <div class="dash-card-title">{card["title"]}</div>
                          <div class="dash-card-desc">{card["desc"]}</div>
                          <div class="dash-features">{badges_html}</div>
                          <div class="dash-input-label">Accepts</div>
                          <div class="dash-input-value">{card["input"]}</div>
                        </div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if st.button(card["label"], key=f"dash_goto_{card['key']}", use_container_width=True):
                    st.session_state["page"] = card["key"]
                    st.query_params["page"] = card["key"]
                    st.rerun()

        # real vertical gap between rows
        if row_start + 2 < len(CARDS):
            st.markdown("<div style='height: 3rem;'></div>", unsafe_allow_html=True)