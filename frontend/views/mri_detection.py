import streamlit as st
from pathlib import Path
import os

def render():
    # Top wrapper to align content
    st.markdown("<div style='max-width:1100px;margin:1.2rem auto;'>", unsafe_allow_html=True)

    # Two-column layout: left for the pill/title (upper-left), right for description and uploader
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.title("MRI Detection")
        st.markdown("Upload a brain MRI file to analyze.")

    with col_right:
        st.markdown(
            """
            <div style='padding:1rem;border-radius:12px;background:#ffffff;border:1px solid #E8F0FF;'>
                <h3 style='margin-top:0;color:#1E3A5F;'>Upload instructions</h3>
                <p style='color:#4B5563;margin-top:0.25rem;'>
                    Please upload a <strong>.nii</strong> or <strong>.nii.gz</strong> file containing a single MRI volume.
                    Files should be anonymized before upload. This frontend only collects the file — connect a backend process to run the model.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        uploaded_file = st.file_uploader(
            "Choose MRI file (.nii / .nii.gz) or an image (jpg/png)",
            type=["nii", "nii.gz", "jpg", "jpeg", "png"],
            help="Select a single .nii, .nii.gz or image file",
        )

        # hard-coded sample image path to show when an image is uploaded
        sample_image_path = os.path.join(os.path.dirname(__file__), "..", "assets", "fcd.jpg")

        if uploaded_file is not None:
            # uploaded_file is a Streamlit UploadedFile object
            name = uploaded_file.name
            try:
                size = uploaded_file.size
            except Exception:
                size = len(uploaded_file.getbuffer())
            size_kb = size / 1024

            st.markdown(
                f"""
                <div style='padding:12px;border-radius:12px;background:#F8FAFF;border:1px solid #E8F0FF; margin-top:0.6rem;'>
                    <strong>Selected file:</strong> {name} <br/>
                    <strong>Size:</strong> {size_kb:.1f} KB
                </div>
                """,
                unsafe_allow_html=True,
            )

            # If the uploaded file is an image type, show the hard-coded image view
            lower_name = name.lower()
            if lower_name.endswith(('.jpg', '.jpeg', '.png')):
                st.success("Image uploaded — showing sample FCD image (frontend-only).")

                # show the hard-coded image if it exists
                img_path = Path(sample_image_path)
                if img_path.exists():
                    st.image(str(img_path), caption="FCD sample image", use_column_width=True)
                else:
                    st.error(f"Sample image not found at: {sample_image_path}")

                # provide a button to proceed (simulated routing to another step)
                if st.button("Proceed"):
                    st.info("Routed to image viewer (frontend placeholder).")
                    # show the image again in a larger container
                    if img_path.exists():
                        st.image(str(img_path), caption="FCD sample image (viewer)", use_column_width=True)

            else:
                # Non-image (assume MRI .nii/.nii.gz) keep existing behavior
                try:
                    file_bytes = uploaded_file.getvalue()
                    st.download_button("Download uploaded file", data=file_bytes, file_name=name)
                except Exception:
                    # fallback: no download offered
                    pass

        # Action buttons under upload area
        cols = st.columns([1, 1])
        with cols[0]:
            analyze_disabled = uploaded_file is None
            analyze = st.button("Analyze MRI", disabled=analyze_disabled)
        with cols[1]:
            st.write("\n")

        if uploaded_file is None:
            st.info("No file selected. Please upload a .nii, .nii.gz or image file to enable analysis.")

        if 'analyze' in locals() and analyze:
            # Frontend placeholder for analysis step
            with st.spinner("Preparing analysis (frontend-only)..."):
                st.success("File received. Analysis backend is not yet wired — implement processing to run the model.")

    st.markdown("</div>", unsafe_allow_html=True)