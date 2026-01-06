import html
import streamlit as st
import streamlit.components.v1 as components
from textwrap import dedent

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


def _clean_cat_keep_placeholder(v: object):
    return v


def _clean_num(v: object):
    return None if v is None else v


def _safe_index(options: list[str], value: object) -> int:
    try:
        return options.index(value)
    except ValueError:
        return 0


def _is_missing_required(v: object) -> bool:
    return _is_placeholder(v)


def _clinician_md_to_html(md: str) -> str:
    """
    Converts clinician_summary markdown-ish text into safe-ish HTML.
    Rules:
      - **Title:** becomes <h4>
      - - item becomes <li>
      - normal lines become <p>
    """
    if not md:
        return ""

    lines = [ln.rstrip() for ln in md.splitlines()]
    out = []
    ul_open = False

    def close_ul():
        nonlocal ul_open
        if ul_open:
            out.append("</ul>")
            ul_open = False

    for ln in lines:
        s = ln.strip()
        if not s:
            close_ul()
            continue

        # IMPORTANT: escape everything so clinician_summary cannot inject HTML
        s = html.escape(s)

        # section headers like **Next checks (practical):**
        if s.startswith("**") and s.endswith("**"):
            close_ul()
            title = s.strip("*").strip().rstrip(":")
            out.append(f"<h4>{title}</h4>")
            continue

        # bullet lines "- something"
        if s.startswith("- "):
            if not ul_open:
                out.append('<ul class="nh-ul">')
                ul_open = True
            item = s[2:].strip().replace("**", "")
            out.append(f"<li>{item}</li>")
            continue

        # normal paragraph
        close_ul()
        p = s.replace("**", "")
        out.append(f'<p class="nh-p">{p}</p>')

    close_ul()
    return "\n".join(out).strip()


def render():
    _init_state()

    MODEL_PATH = "models/asm_response_prediction.pkl"

    try:
        artifacts = get_artifacts(MODEL_PATH)
    except Exception as e:
        st.error(f"Model could not be loaded: {e}")
        return

    # CSS/theme - keep in main page (not iframe)
    st.markdown(
        dedent("""
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
        """).strip(),
        unsafe_allow_html=True,
    )

    left, mid, right = st.columns([2, 28, 2], vertical_alignment="top")

    with mid:
        st.markdown(
            dedent("""
            <div>
              <div class="asm-title">Anti-Seizure Medication Response</div>
              <p class="asm-subtitle">Enter key patient, clinical, and investigation data to estimate seizure freedom at 12 months.</p>
            </div>
            """).strip(),
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
                    value=st.session_state.age_of_onset,
                    placeholder="Enter age at seizure onset", key="age_of_onset"
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
                "age": _clean_num(st.session_state.age),
                "age_of_onset": _clean_num(st.session_state.age_of_onset),
                "pretreatment_seizure_count": _clean_num(st.session_state.pretreatment_seizure_count),
                "prior_asm_exposure_count": _clean_num(st.session_state.prior_asm_exposure_count),

                "sex": _clean_cat_keep_placeholder(st.session_state.sex),
                "seizure_type": _clean_cat_keep_placeholder(st.session_state.seizure_type),
                "current_asm": _clean_cat_keep_placeholder(st.session_state.current_asm),

                "mri_lesion_type": _clean_cat_keep_placeholder(st.session_state.mri_lesion_type),
                "eeg_status_detail": _clean_cat_keep_placeholder(st.session_state.eeg_status_detail),

                "psychiatric_disorder": _clean_cat_keep_placeholder(st.session_state.psychiatric_disorder),
                "intellectual_disability": _clean_cat_keep_placeholder(st.session_state.intellectual_disability),
                "cerebrovascular_disease": _clean_cat_keep_placeholder(st.session_state.cerebrovascular_disease),
                "head_trauma": _clean_cat_keep_placeholder(st.session_state.head_trauma),
                "cns_infection": _clean_cat_keep_placeholder(st.session_state.cns_infection),
                "substance_alcohol_abuse": _clean_cat_keep_placeholder(st.session_state.substance_alcohol_abuse),
                "family_history": _clean_cat_keep_placeholder(st.session_state.family_history),
            }

            required_keys = ["age", "age_of_onset", "pretreatment_seizure_count", "sex", "seizure_type", "current_asm"]
            missing = [k for k in required_keys if _is_missing_required(sample_patient.get(k)) or sample_patient.get(k) is None]
            if missing:
                st.warning(f"Please fill/select required fields: {', '.join(missing)}")
                return

            try:
                out = predict(sample_patient, artifacts)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                return

            result_text = html.escape(out.get("result_text", ""))
            prob = float(out.get("prob_final", 0.0))
            prob_pct = round(prob * 100)

            badge_class = "ok" if out.get("pred_label", 0) == 1 else "bad"
            badge_text = "Likely seizure-free" if out.get("pred_label", 0) == 1 else "Likely not seizure-free"

            summary_html = _clinician_md_to_html(out.get("clinician_summary", ""))

            # Render inside an HTML component so Streamlit cannot show raw tags as code text
            card_html = dedent(f"""
            <html>
              <head>
                <style>
                  /* NOTE: this CSS must be inside the iframe as well */
                  .nh-card{{
                    border: 1px solid rgba(226,232,240,.95);
                    border-radius: 18px;
                    padding: 18px 18px;
                    background: linear-gradient(180deg,#ffffff 0%, #fbfdff 100%);
                    box-shadow: 0 10px 26px rgba(15,23,42,0.08);
                    margin-top: 14px;
                    font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
                  }}
                  .nh-header{{
                    display:flex; align-items:flex-start; justify-content:space-between;
                    gap: 12px; margin-bottom: 10px;
                  }}
                  .nh-title{{
                    font-size: 1.25rem; font-weight: 900; color:#0f172a;
                    line-height: 1.2; margin: 0;
                  }}
                  .nh-badge{{
                    display:inline-flex; align-items:center; gap:8px;
                    padding: 6px 10px; border-radius: 999px;
                    font-weight: 800; font-size: 0.85rem;
                    border: 1px solid rgba(148,163,184,.55);
                    background: rgba(248,250,252,.8);
                    color:#0f172a; white-space: nowrap;
                  }}
                  .nh-badge.ok{{ border-color: rgba(34,197,94,.25); background: rgba(34,197,94,.08); }}
                  .nh-badge.bad{{ border-color: rgba(239,68,68,.25); background: rgba(239,68,68,.08); }}

                  .nh-metrics{{ display:flex; gap: 10px; flex-wrap: wrap; margin: 10px 0 6px 0; }}
                  .nh-metric{{
                    flex: 1 1 160px;
                    border: 1px solid rgba(226,232,240,.95);
                    border-radius: 16px;
                    padding: 12px 12px;
                    background:#ffffff;
                  }}
                  .nh-metric .k{{ font-size: 0.82rem; color:#64748b; font-weight: 700; margin-bottom: 4px; }}
                  .nh-metric .v{{ font-size: 1.15rem; font-weight: 900; color:#0f172a; line-height: 1.1; }}

                  .nh-section{{ margin-top: 14px; border-top: 1px solid rgba(226,232,240,.9); padding-top: 12px; }}
                  .nh-section h4{{ margin: 0 0 8px 0; font-size: 1.02rem; font-weight: 900; color:#1E3A5F; }}
                  .nh-p{{ margin: 0 0 10px 0; color:#0f172a; font-size: 0.95rem; line-height: 1.55; }}
                  .nh-ul{{ margin: 6px 0 10px 0; padding-left: 18px; color:#0f172a; font-size: 0.95rem; line-height: 1.55; }}
                  .nh-ul li{{ margin: 4px 0; }}
                  .nh-foot{{ margin-top: 10px; color:#64748b; font-size: 0.88rem; line-height: 1.45; }}
                </style>
              </head>
              <body>
                <div class="nh-card">
                  <div class="nh-header">
                    <div>
                      <p class="nh-title">Result: {result_text}</p>
                    </div>
                    <div class="nh-badge {badge_class}">{badge_text}</div>
                  </div>

                  <div class="nh-metrics">
                    <div class="nh-metric">
                      <div class="k">Seizure-free probability (12 months)</div>
                      <div class="v">{prob_pct}%</div>
                    </div>
                  </div>

                  <div class="nh-section">
                    <h4>Clinical summary</h4>
                    {summary_html}
                  </div>
                </div>
              </body>
            </html>
            """).strip()

            components.html(card_html, height=1000, scrolling=False)
