import streamlit as st
import requests
import base64
import os
from pathlib import Path
from textwrap import dedent

BACKEND_URL = "http://127.0.0.1:8000"
MRI_ENDPOINT = f"{BACKEND_URL}/epilepsy_diagnosis/mri/predict"

def render():

    # ---------------------------------------
    # STATE (optional reset support like SOZ)
    # ---------------------------------------
    if "mri_uploader_version" not in st.session_state:
        st.session_state["mri_uploader_version"] = 0

    # ---------------------------------------
    # CSS (SOZ-style)
    # ---------------------------------------
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

        div[data-testid="column"] { padding-left: 1rem; padding-right: 1rem; }

        .mri-page { width: 100%; }

        .mri-wrapper { padding: 0rem 0 1rem !important; }
        .nh-title { font-size: 2.5rem; font-weight: 800; color: #1E3A5F; margin-top:-0.6rem !important; }
        .nh-sub { color:#49576B; margin-top:-0.6rem !important; margin-bottom:0.6rem !important; }

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
        div[data-testid="stFileUploader"] {
            border: 1px solid #E6EEF8;
            background: #FAFCFF;
            border-radius: 12px;
            padding: 0.6rem;
            transition: background 160ms ease, box-shadow 160ms ease;
            width: 100%;
        }
        div[data-testid="stFileUploader"]:hover {
            background: #F6FAFF;
            box-shadow: 0 6px 14px rgba(15, 52, 96, 0.07);
        }
        div[data-testid="stFileUploader"] label { margin-bottom: 0 !important; }
        div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
            padding: 0.6rem;
            min-height: 120px;
            display:flex;
            align-items:center;
        }
        div[data-testid="stFileUploader"] button {
            background:#4A7DFF;
            color:#FFFFFF;
            border:none;
            border-radius:10px;
            padding:0.45rem 0.85rem;
            font-weight:700;
        }
        div[data-testid="stFileUploader"] button:hover { background:#356AC3; }
        div[data-testid="stFileUploader"] section { text-align: left; }

        /* Buttons (keep consistent) */
        button[kind="primary"]{
            background:#74B0D3 !important;
            border:1px solid #74B0D3 !important;
            border-radius:14px !important;
            font-weight:700 !important;
        }
        button[kind="primary"]:hover{
            background:#67A3C7 !important;
            border:1px solid #67A3C7 !important;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

    # ---------------------------------------
    # CENTERED LAYOUT (same as ASM/SOZ)
    # ---------------------------------------
    left, mid, right = st.columns([1, 40, 1], vertical_alignment="top")

    with mid:
        st.markdown('<div class="mri-page">', unsafe_allow_html=True)
        st.markdown('<div class="mri-wrapper">', unsafe_allow_html=True)

        # Title + subtitle (like screenshot)
        st.markdown('<div class="nh-title">MRI Detection</div>', unsafe_allow_html=True)
        st.markdown('<div class="nh-sub">Drag and drop an MRI file or browse. Accepted formats: .nii, .nii.gz, .jpg, .png</div>', unsafe_allow_html=True)

        # Main upload card (SOZ structure)
        st.markdown('<div class="card upload-card">', unsafe_allow_html=True)
        st.markdown('<div class="upload-area">', unsafe_allow_html=True)

        info_col, up_col = st.columns([1, 1.2])

        # Left instructions
        with info_col:
            st.markdown(
                '<div class="upload-header">'
                '<div class="upload-icon">🧠</div>'
                '<div class="upload-title">MRI File Upload</div>'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="upload-meta">'
                '<span class="chip">Accepted: NIfTI, JPG/PNG</span>'
                '<span class="chip">Max 500MB</span>'
                '<span class="chip">HIPAA-like demo</span>'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<ul class="list">'
                '<li>Drag a file into the dropzone.</li>'
                '<li>Or click <em>Browse files</em> to select.</li>'
                '<li>Use anonymized MRI files only.</li>'
                '</ul>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="info-card">'
                '<div class="info-title">Expected MRI Format</div>'
                '<ul class="info-list">'
                '<li>NIfTI volume: <strong>.nii</strong> or <strong>.nii.gz</strong></li>'
                '<li>Single 3D brain MRI volume recommended</li>'
                '<li>Optional demo images: <strong>.jpg/.png</strong></li>'
                '</ul>'
                '</div>',
                unsafe_allow_html=True
            )

        # Right uploader (dropzone)
        with up_col:
            uploader_key = f"mri_uploader_{st.session_state['mri_uploader_version']}"

            uploaded_file = st.file_uploader(
                "Upload MRI File",
                type=["nii", "gz", "jpg", "jpeg", "png"],  # "gz" covers .nii.gz
                label_visibility="collapsed",
                accept_multiple_files=False,
                help="Upload .nii / .nii.gz or an image file",
                key=uploader_key,
            )

        st.markdown('</div>', unsafe_allow_html=True)  # upload-area
        st.markdown('</div>', unsafe_allow_html=True)  # card
        st.markdown('</div>', unsafe_allow_html=True)  # wrapper

        # ---------------------------------------
        # POST-UPLOAD UI (kept simple + consistent)
        # ---------------------------------------
        sample_image_path = Path(__file__).resolve().parent.parent / "assets" / "fcd.jpg"

        if uploaded_file is not None:
            name = uploaded_file.name
            lower = name.lower()

            # strict validation (because "gz" also matches random .gz files)
            allowed = (lower.endswith(".nii") or lower.endswith(".nii.gz") or lower.endswith(".jpg") or lower.endswith(".jpeg") or lower.endswith(".png"))
            if not allowed:
                st.error("Unsupported file type. Please upload .nii, .nii.gz, .jpg, .jpeg, or .png.")
                return

            size_kb = (uploaded_file.size / 1024) if hasattr(uploaded_file, "size") else (len(uploaded_file.getvalue()) / 1024)

            st.markdown(
                f"""
                <div class="card" style="margin-top:0.9rem;">
                    <strong>Selected file:</strong> {name}<br/>
                    <strong>Size:</strong> {size_kb:.1f} KB
                </div>
                """,
                unsafe_allow_html=True
            )

            # If image uploaded: show preview (demo)
            if lower.endswith((".jpg", ".jpeg", ".png")):
                st.success("Image uploaded — demo preview.")

                if sample_image_path.exists():
                    st.image(str(sample_image_path), caption="FCD sample image", use_container_width=True)
                else:
                    st.info("Sample preview image not found (assets/fcd.jpg).")

            # Analyze button (primary)
            analyze = st.button("Analyze MRI", type="primary")

            # Backend call
            if analyze:
                with st.spinner("Uploading and analyzing MRI..."):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/octet-stream")}
                        resp = requests.post(MRI_ENDPOINT, files=files, timeout=180)

                        if resp.status_code != 200:
                            st.error(f"Backend error ({resp.status_code}): {resp.text}")
                            return

                        data = resp.json()
                        img_b64 = data.get("image_b64")
                        stats = data.get("stats", {})

                        prediction = stats.get("prediction", "N/A")
                        probability = stats.get("fcd_probability", 0.0)
                        best_slice_info = stats.get("best_slice_info", {})
                        num_patches = stats.get("num_patches", 0)
                        
                        # Define threshold for FCD detection
                        threshold = 0.5
                        fcd_detected = probability >= threshold

                        # Clinical Disclaimer Banner
                        st.markdown(
                            '<div style="background: #FFF3CD; border: 1px solid #FFEAA7; border-radius: 8px; padding: 1rem; margin-bottom: 1.5rem;">'
                            '<div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">'
                            '<span style="color: #856404; font-size: 1.1rem;">⚠️</span>'
                            '<strong style="color: #856404;">Clinical Notice</strong>'
                            '</div>'
                            '<p style="margin: 0; color: #856404; font-size: 0.9rem;">'
                            'This tool is designed for decision support and research use. It does not replace expert radiological or neurological assessment.'
                            '</p>'
                            '</div>',
                            unsafe_allow_html=True
                        )

                        # Clinical Summary Card
                        result_color = "#dc3545" if fcd_detected else "#28a745"
                        result_bg = "#f8d7da" if fcd_detected else "#d4edda"
                        result_border = "#dc3545" if fcd_detected else "#28a745"
                        
                        result_text = "FCD pattern detected by the model" if fcd_detected else "No FCD pattern detected by the model"
                        
                        st.markdown(
                            f'<div style="background: {result_bg}; border: 2px solid {result_border}; border-radius: 12px; padding: 1.2rem; margin-bottom: 1.5rem;">'
                            f'<div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem;">'
                            f'<div style="background: {result_color}; color: white; border-radius: 50%; width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; font-weight: bold;">'
                            f'{"!" if fcd_detected else "✓"}'
                            f'</div>'
                            f'<h3 style="margin: 0; color: {result_color}; font-size: 1.3rem;">{result_text}</h3>'
                            f'</div>'
                            
                            # Probability Bar
                            f'<div style="margin-bottom: 1rem;">'
                            f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">'
                            f'<span style="color: #495057; font-weight: 600;">FCD Probability</span>'
                            f'<span style="color: #495057; font-weight: 700; font-size: 1.1rem;">{probability:.2%}</span>'
                            f'</div>'
                            # Probability bar
                            f'<div style="background: #e9ecef; height: 12px; border-radius: 6px; overflow: hidden; position: relative;">'
                            f'<div style="background: linear-gradient(to right, #28a745, #ffc107, #dc3545); height: 100%; width: {min(probability * 100, 100)}%; border-radius: 6px;"></div>'
                            f'<div style="position: absolute; left: 50%; top: 0; width: 2px; height: 100%; background: #6c757d; transform: translateX(-50%);"></div>'
                            f'</div>'
                            f'<div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #6c757d; margin-top: 0.25rem;">'
                            f'<span>Low (0%)</span>'
                            f'<span style="font-weight: 600;">Threshold (50%)</span>'
                            f'<span>High (100%)</span>'
                            f'</div>'
                            f'</div>'
                            
                            # Analysis Details Grid
                            f'<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; margin-bottom: 1rem; font-size: 0.9rem;">'
                            f'<div><strong>Decision Threshold:</strong> 50%</div>'
                            f'<div><strong>Patches Analyzed:</strong> {num_patches}</div>'
                            f'<div><strong>MRI Slice Reviewed:</strong> {best_slice_info.get("slice_idx", "N/A")}</div>'
                            f'<div><strong>Model Focus Location:</strong> ({best_slice_info.get("x", "N/A")}, {best_slice_info.get("y", "N/A")})</div>'
                            f'</div>'
                            
                            f'<div style="font-size: 0.85rem; color: #6c757d; font-style: italic;">'
                            f'Analysis Type: Patch-based CNN classification with Grad-CAM explanation'
                            f'</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                        # Clinical Interpretation Box
                        if fcd_detected:
                            interpretation = (
                                "The model identified imaging patterns associated with Focal Cortical Dysplasia in the reviewed region. "
                                "The attention map highlights influential regions but is not an exact lesion segmentation. "
                                "Further expert review and correlation with clinical findings is recommended."
                            )
                            interp_color = "#721c24"
                            interp_bg = "#f8d7da"
                        else:
                            interpretation = (
                                "The model did not identify strong imaging patterns consistent with Focal Cortical Dysplasia in the reviewed region. "
                                "The attention map indicates which areas influenced the decision and should not be interpreted as a confirmed abnormality."
                            )
                            interp_color = "#155724"
                            interp_bg = "#d4edda"
                        
                        st.markdown(
                            f'<div style="background: {interp_bg}; border-left: 4px solid {interp_color}; padding: 1rem; margin-bottom: 1.5rem; border-radius: 0 8px 8px 0;">'
                            f'<h4 style="margin: 0 0 0.5rem; color: {interp_color}; font-size: 1rem;">Clinical Interpretation</h4>'
                            f'<p style="margin: 0; color: {interp_color}; font-size: 0.9rem; line-height: 1.5;">{interpretation}</p>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                        # Model Logic Explanation
                        st.markdown(
                            '<div style="background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 1rem; margin-bottom: 1.5rem;">'
                            '<h4 style="margin: 0 0 0.75rem; color: #495057; font-size: 1rem;">Analysis Method</h4>'
                            '<p style="margin: 0; color: #6c757d; font-size: 0.9rem; line-height: 1.5;">'
                            'The model reviews multiple patches across MRI slices and reports the slice with the strongest model response. '
                            'The displayed slice represents the most informative reviewed region, not necessarily a confirmed lesion site. '
                            'Grad-CAM visualization shows model attention patterns during classification.'
                            '</p>'
                            '</div>',
                            unsafe_allow_html=True
                        )

                        mri_b64 = data.get("mri_b64")

                        if img_b64 or mri_b64:
                            # Header for imaging results
                            st.markdown("### Imaging Analysis")
                            
                            col_original, col_overlay = st.columns(2)
                            
                            if mri_b64:
                                with col_original:
                                    mri_bytes = base64.b64decode(mri_b64)
                                    st.image(mri_bytes, caption="MRI slice analyzed by the model", use_container_width=True)
                            
                            if img_b64:
                                with col_overlay:
                                    img_bytes = base64.b64decode(img_b64)
                                    st.image(img_bytes, caption="Grad-CAM attention overlay", use_container_width=True)
                                    
                                    # Helper text about the marker
                                    st.markdown(
                                        '<div style="background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px; padding: 0.6rem; margin-top: 0.5rem;">'
                                        '<p style="margin: 0; color: #495057; font-size: 0.8rem;">'
                                        '<strong>White cross:</strong> Model attention center at patch location ({}, {})'
                                        '</p>'
                                        '</div>'.format(
                                            best_slice_info.get("cx", "N/A"),
                                            best_slice_info.get("cy", "N/A")
                                        ),
                                        unsafe_allow_html=True
                                    )

                            # Enhanced explanation below both images
                            st.markdown(
                                '<div style="background: #e7f3ff; border: 1px solid #b3d9ff; border-radius: 8px; padding: 1rem; margin: 1rem 0;">'
                                '<h4 style="margin: 0 0 0.75rem; color: #004085; font-size: 1rem;">Understanding the Visualization</h4>'
                                '<p style="margin: 0; color: #004085; font-size: 0.9rem; line-height: 1.5;">'
                                'The heatmap highlights regions that influenced the model\'s prediction and '
                                '<strong>does not represent an exact lesion boundary</strong>. '
                                'The overlay shows where the model focused its attention during analysis, '
                                'overlaid directly on the anatomical structures for spatial context. '
                                'Clinical interpretation must consider both the probability score and '
                                'anatomical correlation with expert radiological assessment.'
                                '</p>'
                                '</div>',
                                unsafe_allow_html=True
                            )

                            # Overall interpretation for both images
                            interpretation_style = "background: #f8f9fa; border-left: 4px solid #6c757d; padding: 1rem; margin: 1.5rem 0; border-radius: 0 8px 8px 0;"
                            if fcd_detected:
                                region_text = "region with highest model response"
                                context_text = ("The model identified patterns associated with FCD in the highlighted region. "
                                              "Clinical correlation and expert radiological review are essential for diagnosis.")
                            else:
                                region_text = "areas of model attention"
                                context_text = ("The model did not identify strong FCD patterns. "
                                              "The attention areas show which regions influenced this decision but do not indicate abnormalities.")
                            
                            st.markdown(
                                f'<div style="{interpretation_style}">'
                                f'<h4 style="margin: 0 0 0.5rem; color: #495057; font-size: 0.95rem;">Spatial Correlation</h4>'
                                f'<p style="margin: 0; color: #6c757d; font-size: 0.9rem; line-height: 1.5;">'
                                f'The aligned visualization shows the {region_text} in anatomical context. '
                                f'{context_text} The white marker indicates the center of the analyzed patch '
                                f'for spatial reference on brain hemispheres and anatomical landmarks.'
                                f'</p>'
                                f'</div>',
                                unsafe_allow_html=True
                            )

                        # Collapsible Technical Details
                        with st.expander("🔧 Technical Details (Advanced Users)"):
                            st.markdown("**Raw Model Output:**")
                            st.json(stats)

                    except requests.exceptions.RequestException as e:
                        st.error(f"Could not connect to backend: {e}")

            # Optional reset like SOZ
            if st.button("Upload another MRI", type="secondary"):
                st.session_state["mri_uploader_version"] += 1
                st.rerun()

        else:
            st.info("No file selected. Please upload an MRI (.nii/.nii.gz) or image (.jpg/.png).")

        st.markdown('</div>', unsafe_allow_html=True)  # mri-page