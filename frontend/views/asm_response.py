# views/asm_response.py
import streamlit as st
from utils.asm_predictor import get_artifacts, predict


def _init_state():
    defaults = {
        "asm_show_required": False,

        "age": None,
        "age_of_onset": None,
        "pretreatment_seizure_count": None,
        "prior_asm_exposure_count": None,

        "sex": None,
        "seizure_type": None,
        "current_asm": None,
        "mri_lesion_type": None,
        "eeg_status_detail": None,

        "psychiatric_disorder": None,
        "intellectual_disability": None,
        "cerebrovascular_disease": None,
        "head_trauma": None,
        "cns_infection": None,
        "substance_alcohol_abuse": None,
        "family_history": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def _is_placeholder(v: object) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        s = v.strip().lower()
        return s == "select" or s.startswith("select ")
    return False


def _clean_value(v: object):
    # ✅ key fix: use the same missing token as training pipeline
    return "Unknown" if _is_placeholder(v) else v


def _safe_index(options: list[str], value: object) -> int:
    try:
        return options.index(value)
    except ValueError:
        return 0


def render():
    _init_state()

    MODEL_PATH = "models/asm_response_prediction.pkl"

    try:
        artifacts = get_artifacts(MODEL_PATH)
    except Exception as e:
        st.error(f"Model could not be loaded: {e}")
        return

    st.markdown(
        """
        <style>
        .asm-title { font-size: 2.5rem; font-weight: 800; color:#1E3A5F; }
        .asm-subtitle { color:#49576B; margin-top:-0.4rem; }
        .asm-section-title { font-size: 1.25rem; font-weight: 800; color:#1E3A5F; margin-bottom:0.25rem; }
        .asm-divider { height: 1px; background: rgba(226,232,240,0.9); margin: 0.9rem 0; }
        div[data-testid="column"] { padding-left: 1rem; padding-right: 1rem; }
        div[data-baseweb="input"] > div, div[data-baseweb="select"] > div { border-radius: 12px !important; }
        div[data-testid="stForm"]{ border: 0 !important; outline: 0 !important; box-shadow: none !important; background: transparent !important; padding: 0 !important; margin: 0 !important; }
        input::placeholder { color: #9CA3AF !important; opacity: 1 !important; }
        ul[role="listbox"] li:first-child { color: #9CA3AF !important; }

        div[data-testid="stFormSubmitButton"]{
          display:flex !important;
          justify-content:center !important;
          align-items:center !important;
          margin-left: clamp(0rem, 32vw, 32rem) !important;
          margin-top: 2rem !important;
        }

        div[data-testid="stFormSubmitButton"] button,
        button[data-testid="stBaseButton-primary"],
        button[kind="primary"] {
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
        }

        div[data-testid="stFormSubmitButton"] button:hover,
        button[data-testid="stBaseButton-primary"]:hover,
        button[kind="primary"]:hover {
          background: #67A3C7 !important;
          border: 1px solid #67A3C7 !important;
          color: #ffffff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    left, mid, right = st.columns([2, 28, 2], vertical_alignment="top")

    with mid:
        st.markdown(
            """
            <div>
              <div class="asm-title">Anti-Seizure Medication Response</div>
              <p class="asm-subtitle">Enter key patient, clinical, and investigation data to estimate seizure freedom at 12 months.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        req = " *" if st.session_state.asm_show_required else ""
        SELECT = "Select an option"

        sex_opts = [SELECT, "Female", "Male"]
        seizure_type_opts = [SELECT, "Focal", "Generalized", "Mixed"]
        mri_lesion_opts = [SELECT, "Hippocampal_Sclerosis", "Tumor", "Cortical_Dysplasia"]
        eeg_opts = [SELECT, "Generalized", "Focal", "Normal", "Multifocal"]
        yn_opts = [SELECT, "Yes", "No"]

        trained_asms = artifacts.get("available_asms") or []
        if trained_asms:
            asm_opts = [SELECT] + sorted(list(set(map(str, trained_asms))))
        else:
            asm_opts = [
                SELECT, "Levetiracetam", "Lamotrigine", "Valproate", "Phenobarbital",
                "Carbamazepine", "Phenytoin"
            ]

        with st.form("asm_form_ui_only", clear_on_submit=False):
            st.markdown('<div class="asm-section-title">Clinical Profile</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)

            with c1:
                st.number_input(
                    f"Age (years){req}", min_value=0, max_value=120,
                    value=st.session_state.age, placeholder="Enter age", key="age"
                )
                st.selectbox(
                    f"Seizure Type{req}", seizure_type_opts,
                    index=_safe_index(seizure_type_opts, st.session_state.seizure_type),
                    placeholder="Select seizure type", key="seizure_type"
                )
                st.number_input(
                    f"Number of Seizures Before ASM Initiation{req}",
                    min_value=0, max_value=50,
                    value=st.session_state.pretreatment_seizure_count,
                    placeholder="Enter number of seizures", key="pretreatment_seizure_count"
                )

            with c2:
                st.selectbox(
                    f"Biological Sex (at birth){req}", sex_opts,
                    index=_safe_index(sex_opts, st.session_state.sex),
                    placeholder="Select biological sex", key="sex"
                )
                st.selectbox(
                    "MRI Lesion Type (if available)", mri_lesion_opts,
                    index=_safe_index(mri_lesion_opts, st.session_state.mri_lesion_type),
                    placeholder="Select MRI lesion type", key="mri_lesion_type"
                )
                st.number_input(
                    "Number of Prior Anti-Seizure Medications Tried",
                    min_value=0, max_value=50,
                    value=st.session_state.prior_asm_exposure_count,
                    placeholder="Enter number of prior ASMs", key="prior_asm_exposure_count"
                )

            with c3:
                st.number_input(
                    f"Age at Seizure Onset (years){req}", min_value=0, max_value=120,
                    value=st.session_state.age_of_onset, placeholder="Enter age at seizure onset", key="age_of_onset"
                )
                st.selectbox(
                    "EEG Findings (interictal status)", eeg_opts,
                    index=_safe_index(eeg_opts, st.session_state.eeg_status_detail),
                    placeholder="Select EEG findings", key="eeg_status_detail"
                )
                st.selectbox(
                    f"Current Anti-Seizure Medication{req}", asm_opts,
                    index=_safe_index(asm_opts, st.session_state.current_asm),
                    placeholder="Select current ASM", key="current_asm"
                )

            st.markdown('<div class="asm-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="asm-section-title">Associated Conditions</div>', unsafe_allow_html=True)

            d1, d2, d3 = st.columns(3)
            with d1:
                st.selectbox("Psychiatric Disorders (diagnosed)", yn_opts,
                             index=_safe_index(yn_opts, st.session_state.psychiatric_disorder),
                             placeholder="Select status", key="psychiatric_disorder")
                st.selectbox("History of Head Trauma (with loss of consciousness)", yn_opts,
                             index=_safe_index(yn_opts, st.session_state.head_trauma),
                             placeholder="Select status", key="head_trauma")
                st.selectbox("Family History of Epilepsy", yn_opts,
                             index=_safe_index(yn_opts, st.session_state.family_history),
                             placeholder="Select status", key="family_history")

            with d2:
                st.selectbox("Intellectual Disability (clinical diagnosis)", yn_opts,
                             index=_safe_index(yn_opts, st.session_state.intellectual_disability),
                             placeholder="Select status", key="intellectual_disability")
                st.selectbox("Central Nervous System Infection (e.g., meningitis, encephalitis)", yn_opts,
                             index=_safe_index(yn_opts, st.session_state.cns_infection),
                             placeholder="Select status", key="cns_infection")

            with d3:
                st.selectbox("Cerebrovascular Disease (e.g., stroke, hemorrhage)", yn_opts,
                             index=_safe_index(yn_opts, st.session_state.cerebrovascular_disease),
                             placeholder="Select status", key="cerebrovascular_disease")
                st.selectbox("Substance or Alcohol Use Disorder", yn_opts,
                             index=_safe_index(yn_opts, st.session_state.substance_alcohol_abuse),
                             placeholder="Select status", key="substance_alcohol_abuse")

            submitted = st.form_submit_button("Predict Treatment Response", use_container_width=False)

        if submitted:
            age = st.session_state.age
            onset = st.session_state.age_of_onset
            if age is not None and onset is not None and onset > age:
                st.warning("Age of onset cannot be greater than current age.")
                return

            sample_patient = {
                "age": _clean_value(st.session_state.age),
                "age_of_onset": _clean_value(st.session_state.age_of_onset),
                "pretreatment_seizure_count": _clean_value(st.session_state.pretreatment_seizure_count),
                "prior_asm_exposure_count": _clean_value(st.session_state.prior_asm_exposure_count),

                "sex": _clean_value(st.session_state.sex),
                "seizure_type": _clean_value(st.session_state.seizure_type),
                "current_asm": _clean_value(st.session_state.current_asm),

                "mri_lesion_type": _clean_value(st.session_state.mri_lesion_type),
                "eeg_status_detail": _clean_value(st.session_state.eeg_status_detail),

                "psychiatric_disorder": _clean_value(st.session_state.psychiatric_disorder),
                "intellectual_disability": _clean_value(st.session_state.intellectual_disability),
                "cerebrovascular_disease": _clean_value(st.session_state.cerebrovascular_disease),
                "head_trauma": _clean_value(st.session_state.head_trauma),
                "cns_infection": _clean_value(st.session_state.cns_infection),
                "substance_alcohol_abuse": _clean_value(st.session_state.substance_alcohol_abuse),
                "family_history": _clean_value(st.session_state.family_history),
            }

            required_keys = ["age", "age_of_onset", "pretreatment_seizure_count", "sex", "seizure_type", "current_asm"]
            missing = [k for k in required_keys if sample_patient.get(k) in (None, "Unknown")]
            if missing:
                st.warning(f"Please fill/select required fields: {', '.join(missing)}")
                return

            try:
                pred_label, prob, risk_index, flags = predict(sample_patient, artifacts)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                return

            label_map = {0: "Not seizure-free at 12 months", 1: "Seizure-free at 12 months"}
            pretty = label_map.get(pred_label, str(pred_label))

            st.markdown(f"### Result: **{pretty}**")
            st.markdown(f"### Probability of being seizure-free (12 months): {prob:.2f}")

            st.caption(
                f"Clinical risk index (0–5): {risk_index}  |  "
                f"Model: {artifacts.get('model_name','(unknown)')}  |  "
                f"Threshold: {float(artifacts.get('threshold', 0.5)):.2f}"
            )

            if flags:
                st.warning("Reliability warning: input may be outside typical training support.\n\n- " + "\n- ".join(flags))

            st.caption("Clinical decision support only — does not replace clinician judgment.")
