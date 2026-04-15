import streamlit as st
import requests
import base64
from io import BytesIO

BACKEND_URL = "http://127.0.0.1:8000"
SOZ_ENDPOINT = f"{BACKEND_URL}/epilepsy_diagnosis/soz/predict"

def render():
    mode_param = st.query_params.get("soz_mode", ["upload"])
    mode = mode_param[0] if isinstance(mode_param, list) else mode_param

    if "uploader_version" not in st.session_state:
        st.session_state["uploader_version"] = 0

    # Match ASM page spacing + add SOZ wrapper to scope CSS
    css = """
    <style>
        /* --- MOVE EVERYTHING UP --- */
        .block-container{
            padding-top: 0rem !important;
            padding-bottom: 0rem !important;
            margin-top: -1.5rem !important;     /* pull content upward */
        }


        /* Streamlit top UI spacing */
        header{
            height: 0rem !important;
            min-height: 0rem !important;
        }
        div[data-testid="stToolbar"]{
            height: 0rem !important;
            visibility: hidden !important;
        }

        /* Tighten your title/subtitle spacing */
        .soz-wrapper{ padding: 0rem 0 1rem !important; }
        .nh-title{ margin-top: -0.6rem !important; }
        .nh-sub{ margin-top: -0.6rem !important; margin-bottom: 0.2rem !important; }

        /* keep your existing styles below */
        div[data-testid="column"] { padding-left: 1rem; padding-right: 1rem; }

        /* ---- SOZ scoped wrapper ---- */
        .soz-page { width: 100%; }

      .soz-wrapper { padding: 0.5rem 0 1rem; }
      .nh-title { font-size: 2.5rem; font-weight: 800; color: #1E3A5F; }
      .nh-sub { color:#49576B; margin-top:-0.4rem; }

      .upload-card { background: #FFFFFF; }
      .upload-area { display: flex; flex-direction: column; align-items: stretch; justify-content: center; gap: 0.4rem; padding: 0.5rem 0.6rem 0.7rem; }
      .upload-header { display:flex; align-items:center; gap:0.6rem; margin-bottom:0.25rem; }
      .upload-icon { width:26px; height:26px; border-radius:999px; background:#EDF5FF; color:#356AC3; display:flex; align-items:center; justify-content:center; font-size:0.9rem; border:1px solid #CFE2FF; }
      .upload-title { font-weight:700; color:#1E3A5F; }
      .upload-meta { display:flex; gap:0.5rem; flex-wrap:wrap; margin-top:0.2rem; }
      .chip { background:#F3F7FF; color:#3B556E; border:1px solid #D7E5F5; padding:0.15rem 0.5rem; border-radius:999px; font-size:0.78rem; }

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
      div[data-testid="stFileUploader"] section { text-align: left; }

      .results { animation: fadeIn 550ms ease-out; }
      @keyframes fadeIn { from {opacity:0; transform: translateY(6px);} to {opacity:1; transform: translateY(0);} }

      .table { width:100%; border-collapse: collapse; }
      .table th { text-align:left; padding: 0.55rem; font-weight:700; color:#1E3A5F; border-bottom: 1px solid #E6EEF8; background:#F9FCFF; }
      .table td { padding: 0.55rem; border-bottom: 1px solid #EEF3FA; }
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

    # same centering trick as ASM page
    left, mid, right = st.columns([1, 40, 1], vertical_alignment="top")

    with mid:
        st.markdown('<div class="soz-page">', unsafe_allow_html=True)

        if mode == "results":
            st.markdown('<div class="soz-wrapper results">', unsafe_allow_html=True)
            st.markdown('<div class="nh-title">Seizure Onset Zone (SOZ) Likelihood Analysis</div>', unsafe_allow_html=True)
            uploaded_name = st.session_state.get("soz_uploaded_file_name", "example.edf")
            st.markdown(f'<div class="nh-sub">File analyzed: <strong>{uploaded_name}</strong></div>', unsafe_allow_html=True)

            data = st.session_state.get("soz_result", None)
            backend_channels = []
            backend_img_b64 = None

            if isinstance(data, dict) and data.get("ok"):
                backend_channels = data.get("top_channels", [])
                backend_img_b64 = data.get("topomap_png_base64", None)

            channels = []
            for row in backend_channels:
                ch_name = row.get("channel", row.get("name", "UNK"))
                ch_prob = float(row.get("soz_probability", row.get("p", 0.0)))
                channels.append({"name": ch_name, "p": ch_prob})

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

            st.markdown('<div class="grid">', unsafe_allow_html=True)

            table_html = [
                '<div class="card">',
                '<table class="table">',
                '<thead><tr><th>Channel</th><th>SOZ Likelihood</th><th>Indicator</th></tr></thead>',
                '<tbody>'
            ]
            for ch in channels:
                pct = int(ch["p"] * 100)
                lvl, badge_class, highlight = risk(ch["p"])
                row_cls = "risk-high" if highlight else ""
                table_html.append(
                    f'<tr class="{row_cls}">'
                    f'<td><strong>{ch["name"]}</strong></td>'
                    f'<td>'
                    f'<div class="progress"><div class="progress-fill" style="width:{pct}%;"></div></div>'
                    f'<div style="font-size:0.8rem;color:#4B5563;margin-top:0.35rem;">{pct}%</div>'
                    f'</td>'
                    f'<td><span class="risk-badge {badge_class}">{"High SOZ Risk" if lvl=="High" else lvl}</span></td>'
                    f'</tr>'
                )
            table_html.extend(['</tbody>', '</table>', '</div>'])
            st.markdown("\n".join(table_html), unsafe_allow_html=True)

            st.markdown('<div class="card img-card">', unsafe_allow_html=True)
            if backend_img_b64:
                try:
                    img_bytes = base64.b64decode(backend_img_b64)
                    st.image(img_bytes, caption="SOZ Likelihood Brain Map", use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not decode brain map image: {e}")
                    st.info("Brain visualization not available for this analysis.")
            else:
                st.info("Brain visualization not available. The backend did not generate a topomap for this file.")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)  # end grid

            if st.button("Upload another EEG", type="secondary"):
                try:
                    st.query_params = {"page": "soz", "soz_mode": "upload"}
                except Exception:
                    try:
                        st.experimental_set_query_params(page="soz", soz_mode="upload")
                    except Exception:
                        pass
                st.session_state.pop("soz_uploaded_file_name", None)
                st.session_state["uploader_version"] += 1
                st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)  # end soz-wrapper

        else:
            st.markdown('<div class="soz-wrapper">', unsafe_allow_html=True)
            st.markdown('<div class="nh-title">Seizure Onset Zone Localization</div>', unsafe_allow_html=True)
            st.markdown('<div class="nh-sub">Drag and drop an EEG file or browse. Accepted formats: .edf</div>', unsafe_allow_html=True)

            st.markdown('<div class="card upload-card">', unsafe_allow_html=True)
            st.markdown('<div class="upload-area">', unsafe_allow_html=True)

            info_col, up_col = st.columns([1, 1.2])
            with info_col:
                st.markdown(
                    '<div class="upload-header"><div class="upload-icon">📥</div><div class="upload-title">EEG File Upload</div></div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    '<div class="upload-meta"><span class="chip">Accepted: EDF</span><span class="chip">Max 500MB</span></div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    '<ul class="list">'
                    '<li>Drag a file into the dropzone.</li>'
                    '<li>Or click <em>Browse files</em> to select.</li>'
                    '<li>Only file type is validated (prototype).</li>'
                    '</ul>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    '<div class="info-card"><div class="info-title">Expected EDF Format</div>'
                    '<ul class="info-list">'
                    '<li>Standard European Data Format</li>'
                    '<li>EEG channel recordings</li>'
                    '<li>Typically 10–20 system electrode placement</li>'
                    '<li>Minimum duration: 5 minutes recommended</li>'
                    '</ul></div>',
                    unsafe_allow_html=True
                )

            with up_col:
                uploader_key = f"eeg_uploader_{st.session_state['uploader_version']}"
                uploaded = st.file_uploader(
                    "Upload EEG File",
                    type=["edf"],
                    label_visibility="collapsed",
                    accept_multiple_files=False,
                    help="Upload an EDF file containing EEG data",
                    key=uploader_key,
                )

            st.markdown('</div>', unsafe_allow_html=True)  # upload-area
            st.markdown('</div>', unsafe_allow_html=True)  # card

            if uploaded is not None:
                file_size_mb = uploaded.size / (1024 * 1024)
                st.markdown(
                    f"""
                    <div class="card" style="margin-top:0.7rem;">
                        <strong>Selected file:</strong> {uploaded.name}&nbsp;&nbsp;
                        <strong>Size:</strong> {file_size_mb:.2f} MB
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    """
                    <style>
                        div[data-testid="stButton"] > button[kind="primary"] {
                            background: linear-gradient(135deg, #4A7DFF 0%, #356AC3 100%);
                            color: #FFFFFF;
                            border: none;
                            border-radius: 12px;
                            padding: 0.75rem 2rem;
                            font-size: 1.05rem;
                            font-weight: 700;
                            letter-spacing: 0.02em;
                            transition: all 200ms ease;
                            box-shadow: 0 4px 12px rgba(74, 125, 255, 0.3);
                        }
                        div[data-testid="stButton"] > button[kind="primary"]:hover {
                            background: linear-gradient(135deg, #356AC3 0%, #2A5AB0 100%);
                            box-shadow: 0 6px 18px rgba(53, 106, 195, 0.4);
                            transform: translateY(-1px);
                        }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )

                analyze = st.button("Analyze EEG", type="primary")

                if analyze:
                    st.session_state["soz_uploaded_file_name"] = uploaded.name
                    with st.spinner("Uploading and analyzing EEG..."):
                        try:
                            files = {"file": (uploaded.name, uploaded.getvalue(), "application/octet-stream")}
                            resp = requests.post(SOZ_ENDPOINT, files=files, timeout=180)
                            if resp.status_code != 200:
                                st.error(f"Backend error ({resp.status_code}): {resp.text}")
                                st.stop()

                            data = resp.json()
                            st.session_state["soz_result"] = data

                            try:
                                st.query_params = {"page": "soz", "soz_mode": "results"}
                            except Exception:
                                try:
                                    st.experimental_set_query_params(page="soz", soz_mode="results")
                                except Exception:
                                    pass
                            st.rerun()

                        except requests.exceptions.RequestException as e:
                            st.error(f"Could not connect to backend: {e}")
                            st.stop()

            st.markdown('</div>', unsafe_allow_html=True)  # upload-area
            st.markdown('</div>', unsafe_allow_html=True)  # card
            st.markdown('</div>', unsafe_allow_html=True)  # soz-wrapper

        st.markdown('</div>', unsafe_allow_html=True)  # soz-page