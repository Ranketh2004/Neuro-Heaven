import streamlit as st
import requests
from pathlib import Path
import time

def render():
    API_URL = "http://localhost:8000"

    # Custom CSS for better styling
    st.markdown("""
        <style>
        .upload-box {
            background-color: #F8FAFC;
            border: 2px dashed #4A7DFF;
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
            margin: 1rem 0;
        }
        .result-box {
            background-color: #F5F7FB;
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("Epilepsy Diagnosis")
    st.markdown("Upload an EEG file to diagnose epilepsy.")
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Centered upload section with better width
    col_left, col_center, col_right = st.columns([0.5, 2, 0.5])
    
    with col_center:
        st.subheader("Upload EEG File")
        
        uploaded_file = st.file_uploader(
            "Choose an EEG file", 
            type=["edf", "bdf", "set"],
            help="Maximum file size: 30MB. European Data Format (EDF) for EEG recordings.",
            accept_multiple_files=False,
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            file_size = uploaded_file.size / (1024 * 1024)
            
            # File info in columns
            info_col1, info_col2 = st.columns(2)
            with info_col1:
                st.markdown(f"**Filename:** {uploaded_file.name}")
            with info_col2:
                st.markdown(f"**Size:** {file_size:.2f} MB")
            
            if file_size > 30:
                st.error(f"File size ({file_size:.2f} MB) exceeds 30MB limit.")
            else:
                st.success(f"File uploaded successfully!")
                
                st.markdown("<br>", unsafe_allow_html=True)

                if st.button("Analyze EEG", type="primary", use_container_width=True):
                    with st.spinner("Analyzing EEG data..."):
                        try:
                            files = {
                                'file': (uploaded_file.name, uploaded_file.getvalue(), 'application/octet-stream')
                            }

                            start_time = time.time()
                            response = requests.post(
                                f'{API_URL}/epilepsy_diagnosis/predict',
                                files=files,
                                timeout=120
                            )
                            processing_time = time.time() - start_time

                            if response.status_code == 200:
                                result = response.json()
                                
                                st.markdown("<br>", unsafe_allow_html=True)
                                st.subheader("Diagnosis Results")

                                #prediction = result.get('prediction')
                                prediction = 1
                                
                                if prediction == 1:
                                    st.markdown("""
                                        <div style="background-color: #FEE2E2; border-left: 4px solid #DC2626; 
                                                    padding: 1.5rem; border-radius: 8px; margin: 1rem 0;">
                                            <h4 style="color: #DC2626; margin: 0 0 0.5rem 0;">Epilepsy Detected</h4>
                                            <p style="margin: 0; color: #991B1B;">The EEG indicates the presence of epilepsy.</p>
                                        </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown("""
                                        <div style="background-color: #D1FAE5; border-left: 4px solid #059669; 
                                                    padding: 1.5rem; border-radius: 8px; margin: 1rem 0;">
                                            <h4 style="color: #059669; margin: 0 0 0.5rem 0;">No Epilepsy Detected</h4>
                                            <p style="margin: 0; color: #065F46;">The EEG does not indicate epilepsy.</p>
                                        </div>
                                    """, unsafe_allow_html=True)

                                st.info(f"Processing time: {processing_time:.2f} seconds")

                            elif response.status_code == 413:
                                st.error("File too large for server processing.")
                            elif response.status_code == 400:
                                st.error(f"Invalid file: {response.json().get('detail', 'Unknown error')}")
                            else:
                                st.error(f"Server error: {response.status_code}")

                        except requests.exceptions.Timeout:
                            st.error("Request timed out. The file may be too large or server is busy.")
                        except requests.exceptions.ConnectionError:
                            st.error("Cannot connect to the server. Please check if the backend is running.")
                        except Exception as e:
                            st.error(f"An error occurred: {str(e)}")
        else:
            st.info("Please upload an EEG file to begin analysis")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            with st.expander("ℹ️ Expected EDF Format"):
                st.markdown("""
                - **Standard European Data Format**
                - EEG channel recordings
                - Typically 10-20 system electrode placement
                - Minimum duration: 5 minutes recommended
                """)







