import streamlit as st
import requests
import time
import numpy as np
import plotly.graph_objects as go
from textwrap import dedent

# -----------------------------
# Keep your dashboard function as-is
# -----------------------------
def display_binary_dashboard(predictions, window_size=5):
    st.subheader("Patient Diagnostic Summary")

    total_epochs = len(predictions)
    seizure_epochs = sum(predictions)
    seizure_seconds = seizure_epochs * window_size
    seizure_load = (seizure_epochs / total_epochs) * 100 if total_epochs else 0.0

    is_epileptic = seizure_epochs >= 2  # simple rule

    col1, col2, col3 = st.columns(3)
    with col1:
        color = "inverse" if is_epileptic else "normal"
        st.metric(
            "Patient Status",
            "SEIZURE" if is_epileptic else "NORMAL",
            delta="Alert" if is_epileptic else "Clear",
            delta_color=color
        )
    with col2:
        st.metric("Total Seizure Time", f"{seizure_seconds}s")
    with col3:
        st.metric("Seizure Load", f"{seizure_load:.1f}%", help="Percentage of recording flagged as seizure")

    st.subheader("Temporal Detection Map")

    time_axis = np.arange(len(predictions)) * window_size

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=predictions,
        mode='lines',
        line=dict(color='red', width=2, shape='hv'),
        fill='tozeroy',
        name='AI Detection'
    ))
    fig.add_hrect(y0=0, y1=1, fillcolor="gray", opacity=0.05, layer="below")

    fig.update_layout(
        xaxis_title="Time (Seconds)",
        yaxis=dict(tickvals=[0, 1], ticktext=["Normal", "SEIZURE"], range=[-0.2, 1.2]),
        height=300,
        template="plotly_white",
        margin=dict(l=20, r=20, t=20, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Main page
# -----------------------------
def render():
    API_URL = "http://localhost:8000"
    PREDICT_ENDPOINT = f"{API_URL}/epilepsy_diagnosis/predict"
    MAX_MB = 30

    # -----------------------------
    # State (SOZ pattern)
    # -----------------------------
    mode_param = st.query_params.get("mode", ["upload"])
    mode = mode_param[0] if isinstance(mode_param, list) else mode_param

    if "uploader_version" not in st.session_state:
        st.session_state["uploader_version"] = 0

    # store results in session_state so results page can render
    st.session_state.setdefault("dx_result", None)
    st.session_state.setdefault("dx_processing_time", None)
    st.session_state.setdefault("uploaded_file_name", None)

    # -----------------------------
    # CSS (SOZ pattern)
    # -----------------------------
    css = """
    <style>
        /* --- MOVE EVERYTHING UP --- */
        .block-container{
            padding-top: 0rem !important;
            padding-bottom: 0rem !important;
            margin-top: -1.5rem !important;
        }

        header{
            height: 0rem !important;
            min-height: 0rem !important;
        }
        div[data-testid="stToolbar"]{
            height: 0rem !important;
            visibility: hidden !important;
        }

        /* Tighten title/subtitle spacing */
        .dx-wrapper{ padding: 0rem 0 1rem !important; }
        .nh-title{ margin-top: -0.6rem !important; }
        .nh-sub{ margin-top: -0.6rem !important; margin-bottom: 0.2rem !important; }

        div[data-testid="column"] { padding-left: 1rem; padding-right: 1rem; }

        .dx-page { width: 100%; }

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

        /* Streamlit file uploader */
        div[data-testid="stFileUploader"] { border: 1px solid #E6EEF8; background: #FAFCFF; border-radius: 12px; padding: 0.6rem; transition: background 160ms ease, box-shadow 160ms ease; width: 100%; }
        div[data-testid="stFileUploader"]:hover { background: #F6FAFF; box-shadow: 0 6px 14px rgba(15, 52, 96, 0.07); }
        div[data-testid="stFileUploader"] label { margin-bottom: 0 !important; }
        div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] { padding: 0.6rem; min-height: 120px; display:flex; align-items:center; }
        div[data-testid="stFileUploader"] button { background:#4A7DFF; color:#FFFFFF; border:none; border-radius:10px; padding:0.45rem 0.85rem; font-weight:700; }
        div[data-testid="stFileUploader"] button:hover { background:#356AC3; }
        div[data-testid="stFileUploader"] section { text-align: left; }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

    # centered layout
    left, mid, right = st.columns([1, 40, 1], vertical_alignment="top")

    with mid:
        st.markdown('<div class="dx-page">', unsafe_allow_html=True)

        # -----------------------------
        # RESULTS MODE (SOZ-style)
        # -----------------------------
        if mode == "results":
            st.markdown('<div class="dx-wrapper">', unsafe_allow_html=True)

            st.markdown('<div class="nh-title">Epilepsy Diagnosis</div>', unsafe_allow_html=True)
            fname = st.session_state.get("uploaded_file_name") or "uploaded file"
            st.markdown(f'<div class="nh-sub">File analyzed: <strong>{fname}</strong></div>', unsafe_allow_html=True)

            result = st.session_state.get("dx_result") or {}
            processing_time = st.session_state.get("dx_processing_time")

            prediction = result.get("prediction", None)
            predictions = result.get("predictions", None)

            st.markdown('<div class="card">', unsafe_allow_html=True)

            if prediction == 1:
                st.markdown(
                    """
                    <div style="background-color:#FEE2E2;border-left:4px solid #DC2626;
                                padding:1rem;border-radius:10px;">
                        <h4 style="color:#DC2626;margin:0 0 0.3rem 0;">Epilepsy Detected</h4>
                        <p style="margin:0;color:#991B1B;">The EEG indicates the presence of epilepsy.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            elif prediction == 0:
                st.markdown(
                    """
                    <div style="background-color:#D1FAE5;border-left:4px solid #059669;
                                padding:1rem;border-radius:10px;">
                        <h4 style="color:#059669;margin:0 0 0.3rem 0;">No Epilepsy Detected</h4>
                        <p style="margin:0;color:#065F46;">The EEG does not indicate epilepsy.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.info("No prediction available.")

            if processing_time is not None:
                st.info(f"Processing time: {processing_time:.2f} seconds")

            st.markdown('</div>', unsafe_allow_html=True)  # card

            # Dashboard
            if isinstance(predictions, list) and len(predictions) > 0:
                display_binary_dashboard(predictions, window_size=5)
            else:
                # fallback dummy (only if backend doesn't send predictions)
                dummy = [0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
                display_binary_dashboard(dummy, window_size=5)

            # Back button
            if st.button("Upload another EEG", type="secondary"):
                try:
                    st.query_params = {"page": "epilepsy_diagnosis", "mode": "upload"}
                except Exception:
                    pass
                st.session_state["dx_result"] = None
                st.session_state["dx_processing_time"] = None
                st.session_state["uploaded_file_name"] = None
                st.session_state["uploader_version"] += 1
                st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)  # wrapper
            st.markdown('</div>', unsafe_allow_html=True)  # page
            return

        # -----------------------------
        # UPLOAD MODE (SOZ-style)
        # -----------------------------
        st.markdown('<div class="dx-wrapper">', unsafe_allow_html=True)

        st.markdown('<div class="nh-title">Epilepsy Diagnosis</div>', unsafe_allow_html=True)
        st.markdown('<div class="nh-sub">Drag and drop an EEG file or browse. Accepted formats: .edf, .bdf, .set</div>', unsafe_allow_html=True)

        st.markdown('<div class="card upload-card">', unsafe_allow_html=True)
        st.markdown('<div class="upload-area">', unsafe_allow_html=True)

        info_col, up_col = st.columns([1, 1.2])

        with info_col:
            st.markdown(
                '<div class="upload-header"><div class="upload-icon">📥</div><div class="upload-title">EEG File Upload</div></div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="upload-meta"><span class="chip">Accepted: EDF, BDF, SET</span><span class="chip">Max {MAX_MB}MB</span><span class="chip">HIPAA-like demo</span></div>',
                unsafe_allow_html=True
            )
            st.markdown(
                '<ul class="list">'
                '<li>Drag a file into the dropzone.</li>'
                '<li>Or click <em>Browse files</em> to select.</li>'
                '<li>Files should be anonymized before upload.</li>'
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
            uploaded_file = st.file_uploader(
                "Upload EEG File",
                type=["edf", "bdf", "set"],
                label_visibility="collapsed",
                accept_multiple_files=False,
                help=f"Maximum file size: {MAX_MB}MB",
                key=uploader_key,
            )

        st.markdown('</div>', unsafe_allow_html=True)  # upload-area
        st.markdown('</div>', unsafe_allow_html=True)  # card
        st.markdown('</div>', unsafe_allow_html=True)  # wrapper

        # -----------------------------
        # Actions + file info
        # -----------------------------
        if uploaded_file is None:
            st.info("Please upload an EEG file to begin analysis.")
            st.markdown('</div>', unsafe_allow_html=True)  # page
            return

        # file info + size check
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.markdown(
            f"""
            <div class="card" style="margin-top:0.9rem;">
                <strong>Selected file:</strong> {uploaded_file.name}<br/>
                <strong>Size:</strong> {file_size_mb:.2f} MB
            </div>
            """,
            unsafe_allow_html=True
        )

        if file_size_mb > MAX_MB:
            st.error(f"File size ({file_size_mb:.2f} MB) exceeds {MAX_MB}MB limit.")
            st.markdown('</div>', unsafe_allow_html=True)  # page
            return

        # Buttons (same row)
        c1, c2 = st.columns(2)
        with c1:
            analyze_button = st.button("Analyze EEG", type="primary", use_container_width=True)
        with c2:
            test_button = st.button("Test Visualization", use_container_width=True)

        # Test path: set session + route to results
        if test_button:
            st.session_state["uploaded_file_name"] = uploaded_file.name
            st.session_state["dx_processing_time"] = 0.00
            st.session_state["dx_result"] = {
                "prediction": 1,
                "predictions": [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            }
            try:
                st.query_params = {"page": "epilepsy_diagnosis", "mode": "results"}
            except Exception:
                pass
            st.rerun()

        # Analyze path: call backend, store + route to results
        if analyze_button:
            with st.spinner("Analyzing EEG data..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/octet-stream")}

                    start_time = time.time()
                    response = requests.post(PREDICT_ENDPOINT, files=files, timeout=120)
                    processing_time = time.time() - start_time

                    if response.status_code == 200:
                        st.session_state["uploaded_file_name"] = uploaded_file.name
                        st.session_state["dx_processing_time"] = processing_time
                        st.session_state["dx_result"] = response.json()

                        try:
                            st.query_params = {"page": "epilepsy_diagnosis", "mode": "results"}
                        except Exception:
                            pass
                        st.rerun()

                    elif response.status_code == 413:
                        st.error("File too large for server processing.")
                    elif response.status_code == 400:
                        try:
                            st.error(f"Invalid file: {response.json().get('detail', 'Unknown error')}")
                        except Exception:
                            st.error("Invalid file.")
                    else:
                        st.error(f"Server error: {response.status_code}")

                except requests.exceptions.Timeout:
                    st.error("Request timed out. The file may be too large or server is busy.")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to the server. Please check if the backend is running.")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

        st.markdown('</div>', unsafe_allow_html=True)  # dx-page