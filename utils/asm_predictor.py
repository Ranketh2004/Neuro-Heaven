from __future__ import annotations

from typing import Any, Dict, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st

from utils.model_loader import load_artifacts


@st.cache_resource
def get_artifacts(joblib_path: str) -> Dict[str, Any]:
    return load_artifacts(joblib_path)


def _normalize_text_value(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        if s == "":
            return None
        s_low = s.lower().strip()
        if s_low in {"nan", "null"}:
            return None
        s_low = " ".join(s_low.split())
        s_low = s_low.replace(" ", "_")
        return s_low
    return v


def _normalize_yes_no(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip().lower()
        if s.startswith("select"):
            return None
        if s in {"yes", "y", "true", "1"}:
            return "yes"
        if s in {"no", "n", "false", "0"}:
            return "no"
        return _normalize_text_value(v)
    if isinstance(v, (int, float)) and not np.isnan(v):
        if int(v) == 1:
            return "yes"
        if int(v) == 0:
            return "no"
    return v


def _normalize_value_by_field(field: str, v: Any) -> Any:
    if v is None:
        return None

    comorb_fields = {
        "psychiatric_disorder",
        "intellectual_disability",
        "cerebrovascular_disease",
        "head_trauma",
        "cns_infection",
        "substance_alcohol_abuse",
        "family_history",
    }
    if field in comorb_fields:
        return _normalize_yes_no(v)

    cat_fields = {
        "sex",
        "seizure_type",
        "current_asm",
        "mri_finding",
        "mri_lesion_type",
        "eeg_status_detail",
    }
    if field in cat_fields:
        return _normalize_text_value(v)

    return _normalize_text_value(v)


def build_features(sample_patient: Dict[str, Any], artifacts: Dict[str, Any]) -> pd.DataFrame:
    feature_cols = artifacts["feature_columns_raw"]
    row: Dict[str, Any] = {}
    for col in feature_cols:
        row[col] = _normalize_value_by_field(col, sample_patient.get(col))

    df = pd.DataFrame([row])

    numeric_guess = ["age", "age_of_onset", "pretreatment_seizure_count", "prior_asm_exposure_count"]
    for c in numeric_guess:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def _compute_clinical_risk_index(sample_patient: Dict[str, Any], artifacts: Dict[str, Any]) -> int:
    """
    Mirror your FeatureEngineer risk components (0..5).
    Uses training median seizure count (stored in artifacts if available),
    otherwise uses a conservative fixed cut.
    """
    # fallback cut if not stored: 10
    med_seiz = artifacts.get("guardrails", {}).get("median_seizure_count", None)

    def yn(v):
        v = _normalize_text_value(v)
        return 1 if v in {"yes", "y", "true", "1"} else 0

    # seizure burden
    sct = sample_patient.get("pretreatment_seizure_count")
    try:
        sct = float(sct) if sct is not None else None
    except Exception:
        sct = None

    if med_seiz is None:
        # conservative threshold if unknown
        high_seiz = 1 if (sct is not None and sct >= 10) else 0
    else:
        high_seiz = 1 if (sct is not None and sct > float(med_seiz)) else 0

    # structural lesion
    lesion = _normalize_text_value(sample_patient.get("mri_lesion_type")) or "none"
    structural = 1 if lesion != "none" else 0

    # prior asm
    pasm = sample_patient.get("prior_asm_exposure_count")
    try:
        pasm = float(pasm) if pasm is not None else None
    except Exception:
        pasm = None
    high_prior = 1 if (pasm is not None and pasm > 1) else 0

    # comorbidity (any)
    comorb_fields = [
        "psychiatric_disorder", "intellectual_disability", "cerebrovascular_disease",
        "head_trauma", "cns_infection", "substance_alcohol_abuse", "family_history"
    ]
    any_comorb = 0
    for f in comorb_fields:
        if yn(sample_patient.get(f)) == 1:
            any_comorb = 1
            break

    # eeg abnormal
    eeg = _normalize_text_value(sample_patient.get("eeg_status_detail")) or "normal"
    eeg_abn = 1 if eeg != "normal" else 0

    return int(high_seiz + structural + high_prior + any_comorb + eeg_abn)


def reliability_flags(sample_patient: Dict[str, Any], artifacts: Dict[str, Any]) -> List[str]:
    """
    Lightweight reliability checks:
    - numeric outside [p01, p99]
    - unseen categorical levels
    """
    flags: List[str] = []
    support = artifacts.get("training_support", {}) or {}
    num_ranges = support.get("numeric_ranges", {}) or {}
    cat_levels = support.get("categorical_levels", {}) or {}

    # Numeric checks
    for col in ["age", "age_of_onset", "pretreatment_seizure_count", "prior_asm_exposure_count"]:
        if col not in num_ranges:
            continue
        v = sample_patient.get(col)
        try:
            fv = float(v) if v is not None else None
        except Exception:
            fv = None
        if fv is None:
            continue
        p01 = num_ranges[col].get("p01")
        p99 = num_ranges[col].get("p99")
        if p01 is not None and p99 is not None and (fv < float(p01) or fv > float(p99)):
            flags.append(f"{col} is outside training 1–99% range ({p01:.2f}–{p99:.2f}).")

    # Categorical unseen levels
    for col in ["sex", "seizure_type", "mri_lesion_type", "eeg_status_detail", "current_asm"]:
        if col not in cat_levels:
            continue
        v = _normalize_text_value(sample_patient.get(col))
        if v is None:
            continue
        if v not in set(cat_levels[col]):
            flags.append(f"{col} value '{v}' was not seen in training data.")

    return flags


def apply_clinical_guardrails(sample_patient: Dict[str, Any], artifacts: Dict[str, Any], prob: float) -> Tuple[float, int]:
    """
    More realistic caps to prevent absurd optimism.
    Returns (adjusted_prob, risk_index).
    """
    guard = artifacts.get("guardrails", {}) or {}

    # 1) Global clip (prevents 0.99/0.01 nonsense)
    clip = guard.get("global_prob_clip", {"min": 0.05, "max": 0.95})
    pmin = float(clip.get("min", 0.05))
    pmax = float(clip.get("max", 0.95))
    prob = float(np.clip(prob, pmin, pmax))

    # 2) Compute risk index 0..5
    risk = _compute_clinical_risk_index(sample_patient, artifacts)

    # 3) Risk-based caps
    caps = guard.get("risk_caps", []) or []
    for rule in caps:
        if risk >= int(rule.get("risk_min", 999)):
            cap = float(rule.get("cap", 1.0))
            prob = min(prob, cap)
            break

    # 4) Expanded drug-resistance pattern cap
    crit = guard.get("criteria", {}) or {}
    cap_dr = float(guard.get("extreme_drug_resistance_cap", 0.30))

    seizure_ct = sample_patient.get("pretreatment_seizure_count")
    prior_asms = sample_patient.get("prior_asm_exposure_count")
    lesion = _normalize_text_value(sample_patient.get("mri_lesion_type")) or "none"
    eeg = _normalize_text_value(sample_patient.get("eeg_status_detail")) or "normal"

    try:
        seizure_ct = float(seizure_ct) if seizure_ct is not None else None
    except Exception:
        seizure_ct = None
    try:
        prior_asms = float(prior_asms) if prior_asms is not None else None
    except Exception:
        prior_asms = None

    seizure_cut = float(crit.get("pretreatment_seizure_count_gte", 30))
    prior_cut = float(crit.get("prior_asm_exposure_count_gte", 3))
    lesion_not = str(crit.get("mri_lesion_type_not", "none"))
    eeg_not = str(crit.get("eeg_status_detail_not", "normal"))

    if (
        seizure_ct is not None and seizure_ct >= seizure_cut
        and prior_asms is not None and prior_asms >= prior_cut
        and lesion != lesion_not
        and eeg != eeg_not
    ):
        prob = min(prob, cap_dr)

    return prob, risk


def predict(sample_patient: Dict[str, Any], artifacts: Dict[str, Any]) -> Tuple[int, float, int, List[str]]:
    model = artifacts["model"]
    threshold = float(artifacts["threshold"])

    X = build_features(sample_patient, artifacts)

    if not hasattr(model, "predict_proba"):
        raise ValueError("Loaded model does not support predict_proba().")

    proba = model.predict_proba(X)[0]
    if len(proba) != 2:
        raise ValueError(f"Expected binary classifier with 2 probs, got {len(proba)}")

    prob_pos = float(proba[1])

    # Apply guardrails + compute risk index
    prob_pos, risk = apply_clinical_guardrails(sample_patient, artifacts, prob_pos)

    # Reliability flags
    flags = reliability_flags(sample_patient, artifacts)

    pred = 1 if prob_pos >= threshold else 0
    return pred, prob_pos, risk, flags
