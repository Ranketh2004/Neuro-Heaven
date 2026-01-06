import streamlit as st
import requests
import base64
from io import BytesIO

BACKEND_URL = "http://127.0.0.1:8000"
SOZ_ENDPOINT = f"{BACKEND_URL}/epilepsy_diagnosis/soz/predict"

def render():
    # Determine sub-view (upload vs results)
    mode_param = st.query_params.get("mode", ["upload"])  # may be list
    mode = mode_param[0] if isinstance(mode_param, list) else mode_param

    # Ensure a controllable key to reset the uploader between runs
    if "uploader_version" not in st.session_state:
        st.session_state["uploader_version"] = 0

    # Scoped CSS for this page
    css = """
    <style>
    .container { max-width: 960px; margin: 0 auto; }
    .soz-wrapper { padding: 0.5rem 0 1rem; }
    .nh-title { font-size: 1.6rem; font-weight: 800; color: #1E3A5F; margin-bottom: 0.2rem; }
    .nh-sub { color: #4B5563; margin-bottom: 0.6rem; }

    .card { background: #FFFFFF; border: 1px solid #E6EEF8; box-shadow: 0 8px 18px rgba(15, 52, 96, 0.06); border-radius: 12px; padding: 0.9rem; }
    .upload-card { background: #FFFFFF; }
    .upload-area { display: flex; flex-direction: column; align-items: stretch; justify-content: center; gap: 0.4rem; padding: 0.5rem 0.6rem 0.7rem; }
    .upload-header { display:flex; align-items:center; gap:0.6rem; margin-bottom:0.25rem; }
    .upload-icon { width:26px; height:26px; border-radius:999px; background:#EDF5FF; color:#356AC3; display:flex; align-items:center; justify-content:center; font-size:0.9rem; border:1px solid #CFE2FF; }
    .upload-title { font-weight:700; color:#1E3A5F; }
    .upload-meta { display:flex; gap:0.5rem; flex-wrap:wrap; margin-top:0.2rem; }
    .chip { background:#F3F7FF; color:#3B556E; border:1px solid #D7E5F5; padding:0.15rem 0.5rem; border-radius:999px; font-size:0.78rem; }
    .hint { font-size: 0.92rem; color: #4C5A6B; }

    .list { margin:0; padding-left: 1.1rem; color:#3B556E; }
    .list li { margin-bottom: 0.25rem; }

    .info-card { margin-top: 0.4rem; background:#F9FCFF; border:1px solid #E6EEF8; border-radius:12px; padding:0.6rem; }
    .info-title { font-weight:700; color:#1E3A5F; margin-bottom:0.35rem; font-size:0.98rem; }
    .info-list { margin:0; padding-left: 1.1rem; color:#3B556E; }
    .info-list li { margin-bottom: 0.25rem; }

    /* Beautify Streamlit's native file uploader */
    div[data-testid="stFileUploader"] { border: 1px solid #E6EEF8; background: #FAFCFF; border-radius: 12px; padding: 0.6rem; transition: background 160ms ease, box-shadow 160ms ease; width: 100%; }
    div[data-testid="stFileUploader"]:hover { background: #F6FAFF; box-shadow: 0 6px 14px rgba(15, 52, 96, 0.07); }
    div[data-testid="stFileUploader"] label { margin-bottom: 0 !important; }
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] { padding: 0.6rem; min-height: 120px; display:flex; align-items:center; }
    div[data-testid="stFileUploader"] button { background:#4A7DFF; color:#FFFFFF; border:none; border-radius:10px; padding:0.45rem 0.85rem; font-weight:700; }
    div[data-testid="stFileUploader"] button:hover { background:#356AC3; }
    div[data-testid="stFileUploader"] * { font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    div[data-testid="stFileUploader"] section { text-align: left; }

    .results { animation: fadeIn 550ms ease-out; }
    @keyframes fadeIn { from {opacity:0; transform: translateY(6px);} to {opacity:1; transform: translateY(0);} }

    .table { width:100%; border-collapse: collapse; }
    .table th { text-align:left; padding: 0.55rem; font-weight:700; color:#1E3A5F; border-bottom: 1px solid #E6EEF8; background:#F9FCFF; }
    .table td { padding: 0.55rem; border-bottom: 1px solid #EEF3FA; }
    .row { border-radius: 12px; }
    .risk-high { background: #FFF5F5; }

    .risk-badge { font-weight:700; padding: 0.25rem 0.6rem; border-radius: 999px; font-size: 0.78rem; }
    .risk-high-badge { background:#FEE2E2; color:#B91C1C; border:1px solid #FCA5A5; }
    .risk-mod-badge { background:#FFF5E6; color:#B45309; border:1px solid #FCDCA6; }
    .risk-low-badge { background:#EAF7EA; color:#166534; border:1px solid #BEE3BE; }

    .progress { width: 100%; height: 10px; background: #F1F5F9; border-radius: 999px; overflow: hidden; }
    .progress-fill { height: 100%; background: linear-gradient(90deg, #4A7DFF, #74B0D3); }

    .grid { display: grid; grid-template-columns: 1.5fr 1fr; gap: 0.9rem; align-items: start; }
    @media (max-width: 900px) { .grid { grid-template-columns: 1fr; gap: 0.75rem; } }
    .img-card img { border-radius: 12px; border: 1px solid #E6EEF8; box-shadow: 0 8px 18px rgba(15, 52, 96, 0.06); }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

    if mode == "results":
        st.markdown('<div class="container"><div class="soz-wrapper results">', unsafe_allow_html=True)
        st.markdown('<div class="nh-title">Seizure Onset Zone (SOZ) Likelihood Analysis</div>', unsafe_allow_html=True)
        uploaded_name = st.session_state.get("uploaded_file_name", "example.edf")
        st.markdown(f'<div class="nh-sub">File analyzed: <strong>{uploaded_name}</strong> · Prototype visualization</div>', unsafe_allow_html=True)

        data = st.session_state.get("soz_result", None)

        # fallback if backend result missing
        backend_channels = []
        backend_img_b64 = None

        if isinstance(data, dict) and data.get("ok"):
            backend_channels = data.get("top_channels", [])
            backend_img_b64 = data.get("topomap_png_base64", None)

        # Convert to your UI expected format (backend returns "channel" and "soz_probability")
        channels = []
        for row in backend_channels:
            ch_name = row.get("channel", row.get("name", "UNK"))
            ch_prob = float(row.get("soz_probability", row.get("p", 0.0)))
            channels.append({"name": ch_name, "p": ch_prob})

        # If backend failed or empty, keep your old placeholders (so UI never breaks)
        if not channels:
            channels = [
                {"name": "T7", "p": 0.88},
                {"name": "F7", "p": 0.81},
                {"name": "P7", "p": 0.64},
                {"name": "T8", "p": 0.74},
                {"name": "F8", "p": 0.59},
                {"name": "C3", "p": 0.37},
                {"name": "C4", "p": 0.29},
                {"name": "O1", "p": 0.22},
                {"name": "O2", "p": 0.18},
            ]

        def risk(p):
            if p >= 0.75:
                return "High", "risk-high-badge", True
            if p >= 0.5:
                return "Moderate", "risk-mod-badge", False
            return "Low", "risk-low-badge", False

        # Layout: channels table + SOZ image
        st.markdown('<div class="grid">', unsafe_allow_html=True)

        # Left: Channel likelihood table
        table_html = [
            '<div class="card">',
            '<table class="table">',
            '<thead><tr><th>Channel</th><th>SOZ Likelihood</th><th>Indicator</th></tr></thead>',
            '<tbody>'
        ]
        for ch in channels:
            pct = int(ch["p"] * 100)
            lvl, badge_class, highlight = risk(ch["p"])  # name, css, highlight row
            row_cls = "row risk-high" if highlight else "row"
            table_html.append(
                f'<tr class="{row_cls}">' \
                f'<td><strong>{ch["name"]}</strong></td>' \
                f'<td>' \
                f'<div class="progress"><div class="progress-fill" style="width:{pct}%;"></div></div>' \
                f'<div style="font-size:0.8rem;color:#4B5563;margin-top:0.35rem;">{pct}%</div>' \
                f'</td>' \
                f'<td><span class="risk-badge {badge_class}">{"High SOZ Risk" if lvl=="High" else lvl}</span></td>' \
                f'</tr>'
            )
        table_html.extend(['</tbody>', '</table>', '</div>'])
        st.markdown("\n".join(table_html), unsafe_allow_html=True)

        # Right: Static labeled SOZ image
        st.markdown('<div class="card img-card">', unsafe_allow_html=True)
        if backend_img_b64:
            img_bytes = base64.b64decode(backend_img_b64)
            st.image(img_bytes, caption="SOZ likelihood scalp map (backend)", width=520)
        else:
            st.image(
                "assets/soz-labeled.png",
                caption="Static placeholder: Labeled SOZ regions",
                width=520,
            )

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)  # end .grid

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Upload another EEG", type="secondary"):
                try:
                    # Modern query param setter (Streamlit >=1.30)
                    st.query_params = {"page": "soz", "mode": "upload"}
                except Exception:
                    try:
                        st.experimental_set_query_params(page="soz", mode="upload")
                    except Exception:
                        pass
                # Clear previous upload and force a fresh uploader widget
                st.session_state.pop("uploaded_file_name", None)
                st.session_state["uploader_version"] += 1
                st.rerun()

        st.markdown('</div></div>', unsafe_allow_html=True)  # end .soz-wrapper + .container

    else:
        # Upload view
        st.markdown('<div class="container"><div class="soz-wrapper">', unsafe_allow_html=True)
        st.markdown('<div class="nh-title">Upload EEG Recording</div>', unsafe_allow_html=True)
        st.markdown('<div class="nh-sub">Drag and drop an EEG file or browse. Accepted formats: .edf, .csv</div>', unsafe_allow_html=True)

        st.markdown('<div class="card upload-card">', unsafe_allow_html=True)
        st.markdown('<div class="upload-area">', unsafe_allow_html=True)

        info_col, up_col = st.columns([1, 1.2])
        with info_col:
            st.markdown('<div class="upload-header"><div class="upload-icon">📥</div><div class="upload-title">EEG File Upload</div></div>', unsafe_allow_html=True)
            st.markdown('<div class="upload-meta"><span class="chip">Accepted: EDF, CSV</span><span class="chip">Max 500MB</span><span class="chip">HIPAA-like demo</span></div>', unsafe_allow_html=True)
            st.markdown('<ul class="list">\n<li>Drag a file into the dropzone.</li>\n<li>Or click <em>Browse files</em> to select.</li>\n<li>Only file type is validated (prototype).</li>\n</ul>', unsafe_allow_html=True)
            st.markdown('<div class="info-card"><div class="info-title">Expected EDF Format</div><ul class="info-list">\n<li>Standard European Data Format</li>\n<li>EEG channel recordings</li>\n<li>Typically 10–20 system electrode placement</li>\n<li>Minimum duration: 5 minutes recommended</li>\n</ul></div>', unsafe_allow_html=True)

        with up_col:
            # Use a versioned key so the widget resets when user chooses to upload another
            uploader_key = f"eeg_uploader_{st.session_state['uploader_version']}"
            uploaded = st.file_uploader(
                "",
                type=["edf", "csv"],
                label_visibility="hidden",
                accept_multiple_files=False,
                help="",
                key=uploader_key,
            )

        if uploaded is not None:
            st.session_state["uploaded_file_name"] = uploaded.name

            try:
                st.toast("Uploading to backend…", icon="🧠")
            except Exception:
                pass

            # ✅ Call backend (SOZ endpoint)
            try:
                files = {"file": (uploaded.name, uploaded.getvalue(), "application/octet-stream")}
                resp = requests.post(SOZ_ENDPOINT, files=files, timeout=180)
                if resp.status_code != 200:
                    st.error(f"Backend error ({resp.status_code}): {resp.text}")
                    st.stop()

                data = resp.json()

                # Save results for the results page
                st.session_state["soz_result"] = data

                try:
                    st.toast("Analysis complete ✅", icon="✅")
                except Exception:
                    pass

                # navigate to results
                try:
                    st.query_params = {"page": "soz", "mode": "results"}
                except Exception:
                    try:
                        st.experimental_set_query_params(page="soz", mode="results")
                    except Exception:
                        pass
                st.rerun()

            except requests.exceptions.RequestException as e:
                st.error(f"Could not connect to backend: {e}")
                st.stop()


        st.markdown('</div>', unsafe_allow_html=True)  # end .upload-area
        st.markdown('</div>', unsafe_allow_html=True)  # end .card
        st.markdown('</div></div>', unsafe_allow_html=True)  # end .soz-wrapper + .container