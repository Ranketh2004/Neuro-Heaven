import streamlit as st
from pathlib import Path
import os
import requests
import base64
import io
from textwrap import dedent

def render():
    # CSS/theme - Updated to center the Analyze button horizontally
    st.markdown(
        dedent("""
        <style>
          .asm-title { font-size: 2.5rem; font-weight: 800; color:#1E3A5F; }
          .asm-subtitle { color:#49576B; margin-top:-0.4rem; }
          .asm-section-title { font-size: 1.25rem; font-weight: 800; color:#1E3A5F; margin-bottom:0.25rem; }
          .asm-divider { height: 1px; background: rgba(226,232,240,0.9); margin: 0.9rem 0; }
          
          /* Input and Select styling */
          div[data-testid="column"] { padding-left: 1rem; padding-right: 1rem; }
          div[data-baseweb="input"] > div, div[data-baseweb="select"] > div { border-radius: 12px !important; }
          
          /* Horizontal Centering for the Analyze Button */
          div.stButton {
            display:flex !important;
            justify-content:center !important;
            align-items:center !important;
            margin-left: clamp(0rem, 24vw, 24rem) !important;
          }

          /* Primary Button styling (Matched to File 1) */
          button[data-testid="stBaseButton-primary"],
          button[kind="primary"],
          div[data-testid="stFormSubmitButton"] button {
            width: auto !important;
            padding: 0.85rem 1.75rem !important;
            border-radius: 14px !important;
            font-size: 1.15rem !important;
            font-weight: 700 !important;
            background: #74B0D3 !important;
            border: 1px solid #74B0D3 !important;
            color: #ffffff !important;
            white-space: nowrap !important;
            text-align: center !important;
            transition: background 0.2s ease;
          }

          button[data-testid="stBaseButton-primary"]:hover,
          button[kind="primary"]:hover {
            background: #67A3C7 !important;
            border: 1px solid #67A3C7 !important;
            color: #ffffff !important;
          }
          
          /* Custom container for upload instructions */
          .mri-upload-container {
            padding: 1.5rem;
            border-radius: 18px;
            background: #ffffff;
            border: 1px solid rgba(226,232,240,0.95);
            box-shadow: 0 10px 26px rgba(15,23,42,0.05);
            margin-bottom: 1.5rem;
          }
        </style>
        """).strip(),
        unsafe_allow_html=True,
    )

    # Centered layout matching the screenshot
    left, mid, right = st.columns([2, 28, 2], vertical_alignment="top")

    with mid:
        st.markdown(
            dedent("""
            <div>
              <div class="asm-title">MRI Detection</div>
              <p class="asm-subtitle">Upload a brain MRI file to analyze and detect potential abnormalities.</p>
            </div>
            """).strip(),
            unsafe_allow_html=True,
        )
        
        st.write("") 

        st.markdown(
            """
            <div class="mri-upload-container">
                <div class="asm-section-title">Upload Instructions</div>
                <p style='color:#4B5563; margin-top:0.5rem; line-height:1.5;'>
                    Please upload a <strong>.nii</strong> or <strong>.nii.gz</strong> file containing a single MRI volume.
                    Files should be anonymized before upload. This frontend only collects the file — connect a backend process to run the model.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Choose MRI file (.nii / .nii.gz) or an image (jpg/png)",
            type=None,
            help="Select a single .nii, .nii.gz or image file",
        )

        allowed_exts = ('.nii', '.nii.gz', '.jpg', '.jpeg', '.png')
        sample_image_path = os.path.join(os.path.dirname(__file__), "..", "assets", "fcd.jpg")

        if uploaded_file is not None:
            name = uploaded_file.name
            lower_name = name.lower()
            
            if not lower_name.endswith(allowed_exts):
                st.error("Unsupported file type. Please upload a .nii, .nii.gz or image file.")
                uploaded_file = None
            else:
                try:
                    size = uploaded_file.size
                except Exception:
                    size = len(uploaded_file.getbuffer())
                size_kb = size / 1024

                st.markdown(
                    f"""
                    <div style='padding:12px; border-radius:12px; background:#F8FAFF; border:1px solid #74B0D3; margin-top:0.6rem; color: #1E3A5F;'>
                        <strong>Selected file:</strong> {name} <br/>
                        <strong>Size:</strong> {size_kb:.1f} KB
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if lower_name.endswith(('.jpg', '.jpeg', '.png')):
                    st.success("Image uploaded — showing sample FCD image (frontend-only).")
                    img_path = Path(sample_image_path)
                    if img_path.exists():
                        st.image(str(img_path), caption="FCD sample image", use_column_width=True)
                    
                    if st.button("Proceed", type="primary"):
                        st.info("Routed to image viewer (frontend placeholder).")
                else:
                    try:
                        file_bytes = uploaded_file.getvalue()
                    except Exception:
                        pass

        st.markdown('<div class="asm-divider"></div>', unsafe_allow_html=True)

        # Action Button - Now centered horizontally via CSS
        analyze_disabled = uploaded_file is None
        analyze = st.button("Analyze MRI", disabled=analyze_disabled, type="primary")

        if uploaded_file is None:
            st.info("No file selected. Please upload a .nii, .nii.gz or image file to enable analysis.")

        if analyze:
            with st.spinner("Uploading and analyzing MRI..."):
                try:
                    url = "http://127.0.0.1:8000/epilepsy_diagnosis/mri/predict"
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    resp = requests.post(url, files=files, timeout=120)

                    if resp.status_code != 200:
                        st.error(f"Backend error: {resp.status_code} - {resp.text}")
                    else:
                        data = resp.json()
                        img_b64 = data.get("image_b64")
                        stats = data.get("stats")

                        if img_b64:
                            img_bytes = base64.b64decode(img_b64)
                            st.image(img_bytes, caption="FCD overlay (backend)", use_column_width=True)

                        if stats:
                            st.markdown('<div class="asm-section-title">Prediction Stats</div>', unsafe_allow_html=True)
                            st.json(stats)

                except Exception as e:
                    st.error(f"Failed to call backend: {e}")