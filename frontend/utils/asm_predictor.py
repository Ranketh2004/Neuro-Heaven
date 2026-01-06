# frontend/utils/asm_predictor.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, List, Optional

import numpy as np
import pandas as pd

from utils.pickle_shims import register_pickle_shims

STRICT_VERSION_CHECK = False  # True = hard-fail on mismatch
DEFAULT_THRESHOLD = 0.50

# SHAP settings
ENABLE_SHAP = True
SHAP_TOP_N = 6  # total contributions shown (split into supports vs against)
SHAP_MAX_FEATURES_IN_TEXT = 5  # cap per side in narrative
SHAP_BACKGROUND_N = 50  # only used if we have to build a background (rare)

# Raw categorical columns used by UI 
_CAT_BASE_COLS = {
    "sex",
    "seizure_type",
    "current_asm",
    "mri_lesion_type",
    "eeg_status_detail",
    "psychiatric_disorder",
    "intellectual_disability",
    "cerebrovascular_disease",
    "head_trauma",
    "cns_infection",
    "substance_alcohol_abuse",
    "family_history",
}

# Friendly names for common engineered/base fields (used when SHAP names are ugly)
FRIENDLY_NAMES = {
    "age": "Current age",
    "age_of_onset": "Age at seizure onset",
    "duration_since_onset": "Duration since onset",
    "pretreatment_seizure_count": "Pre-treatment seizure count",
    "seizure_burden_log": "Overall seizure burden (log)",
    "seizure_frequency_risk": "Seizure frequency relative to age",
    "prior_asm_exposure_count": "Number of prior ASMs tried",
    "high_prior_asm": "More than one prior ASM",
    "poly_asm_history": "Three or more prior ASMs",
    "structural_lesion_flag": "Structural lesion on MRI",
    "mri_lesion_type": "MRI lesion type",
    "eeg_epileptic_flag": "Epileptiform EEG activity",
    "eeg_status_detail": "EEG status",
    "comorbidity_burden": "Comorbidity burden",
    "clinical_risk_index": "Clinical risk index (engineered)",
    "sex": "Biological sex",
    "seizure_type": "Seizure type",
    "current_asm": "Current ASM",
}

# ============================================================
# Version helpers (artifact -> runtime guard)
# ============================================================
def _get_versions_runtime() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    try:
        import sklearn
        versions["sklearn"] = str(sklearn.__version__)
    except Exception:
        versions["sklearn"] = "unknown"

    try:
        versions["numpy"] = str(np.__version__)
    except Exception:
        versions["numpy"] = "unknown"

    try:
        versions["pandas"] = str(pd.__version__)
    except Exception:
        versions["pandas"] = "unknown"

    try:
        import lightgbm
        versions["lightgbm"] = str(lightgbm.__version__)
    except Exception:
        versions["lightgbm"] = "unknown"

    return versions


def _version_guard(obj: Dict[str, Any]) -> None:
    trained_versions = obj.get("versions")
    runtime_versions = _get_versions_runtime()

    if trained_versions is None:
        msg = (
            "Artifact does not contain 'versions'. "
            "To guarantee notebook/UI consistency, re-export the model with versions stored."
        )
        if STRICT_VERSION_CHECK:
            raise RuntimeError(msg)
        print("[WARN]", msg)
        return

    diffs = []
    for k in ("sklearn", "numpy", "pandas", "lightgbm"):
        t = str(trained_versions.get(k, "unknown"))
        r = str(runtime_versions.get(k, "unknown"))
        if t != "unknown" and r != "unknown" and t != r:
            diffs.append(f"{k}: trained={t} runtime={r}")

    if diffs:
        msg = "Version mismatch detected: " + " | ".join(diffs)
        if STRICT_VERSION_CHECK:
            raise RuntimeError(msg)
        print("[WARN]", msg)


def _ensure_predict_proba(model: Any) -> Any:
    if model is None:
        raise ValueError("Artifacts missing 'model'.")
    if not hasattr(model, "predict_proba"):
        raise TypeError("Loaded model does not support predict_proba().")
    return model


# ============================================================
# Artifact loading
# ============================================================
def get_artifacts(model_path: str) -> Dict[str, Any]:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact file not found: {path.resolve()}")

    register_pickle_shims()

    obj: Any = None
    errors: List[str] = []

    try:
        import joblib
        obj = joblib.load(path)
    except Exception as e:
        errors.append(f"joblib.load failed: {e}")

    if obj is None:
        try:
            import pickle
            with open(path, "rb") as f:
                obj = pickle.load(f)
        except Exception as e:
            errors.append(f"pickle.load failed: {e}")

    if obj is None:
        raise RuntimeError("Model could not be loaded. " + " | ".join(errors))

    if not isinstance(obj, dict):
        obj = {"model": obj}

    if obj.get("threshold") is None:
        obj["threshold"] = DEFAULT_THRESHOLD

    if not obj.get("model_name"):
        obj["model_name"] = obj.get("best_base_model") or type(obj.get("model")).__name__

    obj.setdefault("available_asms", [])
    obj.setdefault("feature_columns_raw", None)

    _version_guard(obj)
    obj["model"] = _ensure_predict_proba(obj.get("model"))
    return obj


# ============================================================
# Notebook-consistent helpers
# ============================================================
def _norm_str(v: Any) -> str:
    return "" if v is None else str(v).strip().lower()


def _norm_yes(v: Any) -> bool:
    return _norm_str(v) in {"yes", "y", "true", "1"}


def _align_columns_like_notebook(df_one: pd.DataFrame, training_cols: Optional[List[str]]) -> pd.DataFrame:
    if not training_cols:
        return df_one

    df_one = df_one.copy()
    for c in training_cols:
        if c not in df_one.columns:
            df_one[c] = "unknown" if c in _CAT_BASE_COLS else 0

    return df_one[training_cols].copy()


def _coerce_numeric(df_one: pd.DataFrame) -> pd.DataFrame:
    df_one = df_one.copy()
    for c in ["age", "age_of_onset", "pretreatment_seizure_count", "prior_asm_exposure_count"]:
        if c in df_one.columns:
            df_one[c] = pd.to_numeric(df_one[c], errors="coerce")
    return df_one


# ============================================================
# Internals: find fitted pipeline inside calibrated models
# ============================================================
def _extract_fitted_pipeline_and_estimator(model: Any) -> Tuple[Optional[Any], Optional[Any]]:
    """
    Returns (pipeline, final_estimator) if discoverable.
    Works for:
      - Pipeline directly
      - CalibratedClassifierCV wrapping estimator/pipeline (sklearn versions vary)
    """
    # If it's a Pipeline
    if hasattr(model, "named_steps"):
        pipe = model
        est = getattr(pipe, "steps", [])[-1][1] if getattr(pipe, "steps", None) else None
        return pipe, est

    # If it's CalibratedClassifierCV
    try:
        cc0 = model.calibrated_classifiers_[0]
        base = getattr(cc0, "estimator", None) or getattr(cc0, "base_estimator", None)
        if base is not None:
            if hasattr(base, "named_steps"):
                pipe = base
                est = getattr(pipe, "steps", [])[-1][1] if getattr(pipe, "steps", None) else None
                return pipe, est
            return None, base
    except Exception:
        pass

    return None, None


def _transform_to_model_matrix(pipe: Any, raw_df: pd.DataFrame) -> Tuple[np.ndarray, Optional[List[str]]]:
    """
    Apply pipe steps up to the final estimator, returning numeric matrix + names.
    Assumes steps like valclean, fe, preprocess (ColumnTransformer).
    """
    X = raw_df.copy()

    if hasattr(pipe, "named_steps"):
        # Apply feature steps if present
        for step_name in ("valclean", "fe"):
            if step_name in pipe.named_steps:
                X = pipe.named_steps[step_name].transform(X)

        preprocess = pipe.named_steps.get("preprocess")
        if preprocess is not None:
            Xt = preprocess.transform(X)
            names = None
            try:
                names = [str(n) for n in preprocess.get_feature_names_out()]
            except Exception:
                names = None
            return Xt, names

    return X.to_numpy(), list(X.columns)


# ============================================================
# SHAP: compute top contributions + clinician-friendly text
# ============================================================
def _pretty_feature_name(name: str) -> str:
    """
    Turns model feature names into readable labels.
    Handles:
      - prefixes like num__/cat__
      - one-hot like mri_lesion_type_hippocampal_sclerosis
      - base keys that contain underscores (age_of_onset)
    """
    s = str(name)

    for prefix in ("num__", "cat__", "remainder__"):
        if s.startswith(prefix):
            s = s[len(prefix):]

    keys_sorted = sorted(FRIENDLY_NAMES.keys(), key=len, reverse=True)
    for base in keys_sorted:
        if s == base:
            return FRIENDLY_NAMES.get(base, base)
        if s.startswith(base + "_"):
            rest = s[len(base) + 1:]
            base_name = FRIENDLY_NAMES.get(base, base.replace("_", " ").title())
            return f"{base_name}: {rest.replace('_', ' ').title()}"

    return s.replace("_", " ").strip().title()


def _value_str(patient: Dict[str, Any], key: str) -> Optional[str]:
    v = patient.get(key)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return str(v)


def _clinical_phrase_from_feature(feature_label: str, patient: Dict[str, Any]) -> str:
    fl = feature_label.lower()

    if "pre-treatment seizure" in fl or "pretreatment seizure" in fl:
        sc = _value_str(patient, "pretreatment_seizure_count")
        return f"Seizure burden before treatment (reported {sc})"

    if "number of prior asms" in fl or "prior asms" in fl:
        pa = _value_str(patient, "prior_asm_exposure_count")
        return f"History of prior anti-seizure medications tried (reported {pa})"

    if "age at seizure onset" in fl:
        ao = _value_str(patient, "age_of_onset")
        return f"Age at seizure onset (reported {ao})"

    if fl == "current age" or "current age" in fl:
        age = _value_str(patient, "age")
        return f"Current age (reported {age})"

    if "duration since onset" in fl:
        try:
            age = float(patient.get("age"))
            onset = float(patient.get("age_of_onset"))
            dur = max(age - onset, 0)
            return f"Epilepsy duration (approx. {dur:.0f} years)"
        except Exception:
            return "Epilepsy duration"

    if "seizure frequency relative to age" in fl or "seizure_frequency_risk" in fl:
        return "Seizure burden relative to age (higher burden generally worsens prognosis)"

    if "overall seizure burden" in fl or "seizure_burden_log" in fl:
        return "Overall seizure burden (higher burden generally worsens prognosis)"

    if "eeg" in fl:
        eeg = _value_str(patient, "eeg_status_detail")
        return f"EEG status (reported {eeg})"

    if "mri lesion type" in fl or "structural lesion" in fl:
        mri = _value_str(patient, "mri_lesion_type")
        if mri and str(mri).strip().lower() not in {"select", "select an option"}:
            return f"MRI lesion type (reported {mri})"
        return "MRI lesion information"

    if "mri lesion type:" in feature_label:
        return feature_label.replace("MRI lesion type:", "MRI finding:")

    if "seizure type:" in feature_label:
        return feature_label.replace("Seizure type:", "Seizure type:")

    return feature_label


def _summarize_shap_for_doctor(shap_rows: List[Dict[str, Any]], patient: Dict[str, Any]) -> Dict[str, Any]:
    shap_rows = sorted(shap_rows, key=lambda r: abs(float(r.get("shap", 0.0))), reverse=True)

    supports: List[str] = []
    against: List[str] = []
    seen = set()

    for r in shap_rows:
        feat = r.get("feature", "")
        direction = r.get("direction", "")
        label = _clinical_phrase_from_feature(feat, patient)

        key = label.lower()
        if key in seen:
            continue
        seen.add(key)

        if direction == "supports":
            supports.append(label)
        else:
            against.append(label)

    return {
        "supports": supports[:SHAP_MAX_FEATURES_IN_TEXT],
        "against": against[:SHAP_MAX_FEATURES_IN_TEXT],
    }


def _compute_shap_explanation(model: Any, df_one: pd.DataFrame, patient: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns:
      {
        "ok": bool,
        "top": [{feature, shap, direction}, ...],
        "doctor": {supports: [...], against: [...]},
        "note": str
      }
    """
    if not ENABLE_SHAP:
        return {"ok": False, "note": "SHAP disabled."}

    try:
        import shap
    except Exception:
        return {"ok": False, "note": "SHAP not installed in runtime environment."}

    pipe, est = _extract_fitted_pipeline_and_estimator(model)
    if pipe is None or est is None:
        return {"ok": False, "note": "Could not locate fitted pipeline/estimator for SHAP."}

    Xt, feature_names = _transform_to_model_matrix(pipe, df_one)

    try:
        import scipy.sparse as sp
        Xt_dense = Xt.toarray() if sp.issparse(Xt) else np.asarray(Xt)
    except Exception:
        Xt_dense = np.asarray(Xt)

    try:
        explainer = shap.TreeExplainer(est)
        shap_values = explainer.shap_values(Xt_dense)
    except Exception:
        try:
            bg = Xt_dense
            explainer = shap.KernelExplainer(est.predict_proba, bg)
            shap_values = explainer.shap_values(Xt_dense, nsamples=100)
        except Exception as e:
            return {"ok": False, "note": f"SHAP failed: {e}"}

    if isinstance(shap_values, list) and len(shap_values) >= 2:
        sv = np.asarray(shap_values[1])[0]
    else:
        sv = np.asarray(shap_values)[0]

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(sv))]

    idx = np.argsort(np.abs(sv))[::-1][:SHAP_TOP_N]
    rows: List[Dict[str, Any]] = []
    for i in idx:
        val = float(sv[i])
        direction = "supports" if val > 0 else "against"
        rows.append({
            "feature": _pretty_feature_name(feature_names[i]),
            "shap": val,
            "direction": direction,
        })

    doctor = _summarize_shap_for_doctor(rows, patient=patient)

    return {
        "ok": True,
        "top": rows,
        "doctor": doctor,
        # "note": "SHAP explains the trained model’s base prediction (before medication rule adjustment).",
    }


# ============================================================
# ASM rule layer (internal)
# ============================================================
def apply_asm_rules(patient: Dict[str, Any], asm: str, base_prob: float) -> Tuple[float, float, List[str]]:
    asm = _norm_str(asm)
    sex = _norm_str(patient.get("sex"))
    seizure_type = _norm_str(patient.get("seizure_type"))

    age = patient.get("age", None)
    try:
        age = float(age) if age is not None else None
    except Exception:
        age = None

    psych = _norm_yes(patient.get("psychiatric_disorder"))
    int_dis = _norm_yes(patient.get("intellectual_disability"))
    cerebro = _norm_yes(patient.get("cerebrovascular_disease"))
    head_trauma = _norm_yes(patient.get("head_trauma"))
    cns_inf = _norm_yes(patient.get("cns_infection"))
    alcohol = _norm_yes(patient.get("substance_alcohol_abuse"))

    penalty = 0.0
    reasons: List[str] = []

    if asm == "valproate":
        if sex == "female" and (age is not None and age < 50):
            penalty += 0.25
            reasons.append("Valproate: caution in females of childbearing potential (teratogenic risk).")
        if seizure_type == "generalized":
            penalty -= 0.05
            reasons.append("Valproate: can be effective in generalized epilepsy (context-dependent).")
        if alcohol:
            penalty += 0.05
            reasons.append("Valproate: substance/alcohol use may complicate adherence and hepatic monitoring.")
        if int_dis:
            penalty += 0.03
            reasons.append("Valproate: monitor sedation/cognitive effects if vulnerable.")

    if asm == "levetiracetam":
        if psych:
            penalty += 0.12
            reasons.append("Levetiracetam: may worsen irritability/mood in some patients; monitor psychiatric symptoms.")
        if head_trauma or cns_inf:
            penalty -= 0.03
            reasons.append("Levetiracetam: broad-spectrum option; often used in structural/acquired epilepsy contexts.")
        if alcohol:
            penalty += 0.03
            reasons.append("Levetiracetam: substance/alcohol use may increase behavioral risk and reduce adherence.")

    if asm == "lamotrigine":
        if psych:
            penalty -= 0.05
            reasons.append("Lamotrigine: often mood-neutral/mood-friendly; monitor response.")
        try:
            sc = float(patient.get("pretreatment_seizure_count", 0))
        except Exception:
            sc = 0.0
        if sc >= 30:
            penalty += 0.05
            reasons.append("Lamotrigine: slow titration may delay control in high-burden cases.")

    if asm == "carbamazepine":
        if seizure_type == "generalized":
            penalty += 0.18
            reasons.append("Carbamazepine: avoid in generalized epilepsy due to potential seizure worsening.")
        if alcohol:
            penalty += 0.05
            reasons.append("Carbamazepine: interactions/adherence concerns; review co-medications.")
        if cerebro:
            penalty += 0.06
            reasons.append("Carbamazepine: caution with cerebrovascular disease; review tolerability/interactions.")
        if age is not None and age >= 65:
            penalty += 0.06
            reasons.append("Carbamazepine: elderly—higher tolerability and hyponatremia risk.")

    if asm == "phenobarbital":
        if int_dis:
            penalty += 0.18
            reasons.append("Phenobarbital: sedation/cognitive effects may be problematic; consider alternatives if feasible.")
        if psych:
            penalty += 0.08
            reasons.append("Phenobarbital: may worsen mood/behavior; monitor psychiatric symptoms.")
        if age is not None and age >= 65:
            penalty += 0.18
            reasons.append("Phenobarbital: elderly—sedation and falls risk; avoid if possible.")
        if cerebro:
            penalty += 0.10
            reasons.append("Phenobarbital: caution in cerebrovascular disease; sedation/falls concern.")
        if alcohol:
            penalty += 0.08
            reasons.append("Phenobarbital: avoid with alcohol/sedatives (CNS/respiratory depression risk).")

    if asm == "phenytoin":
        if age is not None and age >= 65:
            penalty += 0.12
            reasons.append("Phenytoin: elderly—higher toxicity risk due to PK variability and interactions.")
        if cerebro:
            penalty += 0.08
            reasons.append("Phenytoin: caution in cerebrovascular disease; interactions/side-effects.")
        if alcohol:
            penalty += 0.06
            reasons.append("Phenytoin: alcohol can alter levels and adherence; monitor if used.")
        if int_dis:
            penalty += 0.06
            reasons.append("Phenytoin: cognitive side-effects possible; monitor function.")

    penalty = max(penalty, -0.10)
    adjusted = float(np.clip(base_prob - penalty, 0.0, 0.95))
    return adjusted, float(penalty), reasons


# ============================================================
# Reliability / Applicability
# ============================================================
def _reliability_flags(x: Dict[str, Any]) -> List[str]:
    flags: List[str] = []

    def _to_float(v):
        try:
            return float(v)
        except Exception:
            return None

    age = _to_float(x.get("age"))
    onset = _to_float(x.get("age_of_onset"))
    ptsc = _to_float(x.get("pretreatment_seizure_count"))
    prior = _to_float(x.get("prior_asm_exposure_count"))

    if age is not None and (age < 0 or age > 120):
        flags.append("Age outside plausible range (0–120).")

    if age is not None and onset is not None and onset > age:
        flags.append("Age of onset greater than current age (inconsistent).")

    if ptsc is not None and ptsc > 500:
        flags.append("Pre-treatment seizure count extremely high; probability may be less reliable.")

    if prior is not None and prior > 50:
        flags.append("Prior ASM exposure unusually high; probability may be less reliable.")

    return flags


def _applicability_indicator(x: Dict[str, Any]) -> int:
    score = 0

    def _to_float(v):
        try:
            return float(v)
        except Exception:
            return None

    ptsc = _to_float(x.get("pretreatment_seizure_count"))
    if ptsc is not None:
        if ptsc >= 20:
            score += 2
        elif ptsc >= 10:
            score += 1

    prior = _to_float(x.get("prior_asm_exposure_count"))
    if prior is not None:
        if prior >= 3:
            score += 2
        elif prior >= 1:
            score += 1

    eeg = (x.get("eeg_status_detail") or "").strip().lower()
    if eeg in {"focal", "generalized", "multifocal"}:
        score += 1

    mri = x.get("mri_lesion_type")
    if mri and str(mri).strip().lower() not in {"select", "select an option"}:
        score += 1

    return int(min(5, max(0, score)))


# ============================================================
# Clinician-friendly narrative output (+ SHAP)
# ============================================================
def _prob_band(prob: float) -> str:
    p = float(prob)
    if p >= 0.80:
        return "High"
    if p >= 0.60:
        return "Moderate–high"
    if p >= 0.40:
        return "Intermediate"
    return "Low"


def _format_percent(prob: float) -> str:
    return f"{float(prob) * 100:.0f}%"


def _clean_display(v: Any) -> str:
    if v is None:
        return "Not provided"
    if isinstance(v, float) and np.isnan(v):
        return "Not provided"
    s = str(v).strip()
    if s == "":
        return "Not provided"
    if s.lower() in {"select", "select an option"}:
        return "Not provided"
    return s


def _duration_years(patient: Dict[str, Any]) -> Optional[int]:
    try:
        age = float(patient.get("age"))
        onset = float(patient.get("age_of_onset"))
        if np.isnan(age) or np.isnan(onset):
            return None
        return int(max(age - onset, 0))
    except Exception:
        return None


def _clinician_summary(
    pretty_label: str,
    prob_final: float,
    threshold: float,
    patient: Dict[str, Any],
    reliability_flags: List[str],
    asm_notes: List[str],
    shap_doctor: Optional[Dict[str, Any]] = None,
    shap_note: str = "",
) -> str:
    band = _prob_band(prob_final)
    pct = _format_percent(prob_final)
    likely = prob_final >= threshold

    dur = _duration_years(patient)
    _ = dur  # kept for future expansion

    if likely:
        impression = (
            f"Estimated probability of **seizure freedom at 12 months = {pct}** "
            f"(**{band} likelihood**). This leans toward seizure freedom, but uncertainty remains."
        )
    else:
        impression = (
            f"Estimated probability of **seizure freedom at 12 months = {pct}** "
            f"(**{band} likelihood**). This leans away from seizure freedom without optimization."
        )

    supports, against = [], []
    if shap_doctor:
        supports = shap_doctor.get("supports", [])[:SHAP_MAX_FEATURES_IN_TEXT]
        against = shap_doctor.get("against", [])[:SHAP_MAX_FEATURES_IN_TEXT]

    lines: List[str] = []
    lines.append(impression)
    lines.append("")

    # IMPORTANT: single-level bullets only (your HTML converter does not support nested bullets)
    if shap_doctor and (supports or against):
        # lines.append("**Key drivers (model associations, not causality):**")
        if supports:
            lines.append("**Factors that increased the estimate:**")
            for x in supports:
                lines.append(f"- {x}")
        if against:
            lines.append("**Factors that reduced the estimate:**")
            for x in against:
                lines.append(f"- {x}")
        if shap_note:
            lines.append("")
            lines.append(f"*Note:* {shap_note}")
        lines.append("")

    if asm_notes:
        lines.append("**Medication note:**")
        for n in asm_notes[:3]:
            lines.append(f"- {n}")
        lines.append("")

    if reliability_flags:
        lines.append("**Data quality / applicability flags:**")
        for f in reliability_flags[:2]:
            lines.append(f"- {f}")
        lines.append("")

    lines.append("**Next checks :**")
    lines.append("- Confirm seizure count timeframe and medication adherence.")
    lines.append("- Ensure EEG/MRI are formally reviewed and correlate with semiology.")
    lines.append("- Reassess at follow-up with early response (first 4–12 weeks often changes prognosis).")
    lines.append("")

    return "\n".join(lines)


# ============================================================
# Primary prediction
# ============================================================
def predict_patient(sample_patient: Dict[str, Any], artifacts: Dict[str, Any], use_rules: bool = True) -> Dict[str, Any]:
    model = _ensure_predict_proba(artifacts.get("model"))
    threshold = float(artifacts.get("threshold", DEFAULT_THRESHOLD))

    df_one = pd.DataFrame([{k: (np.nan if v is None else v) for k, v in sample_patient.items()}])

    cols = artifacts.get("feature_columns_raw")
    df_one = _align_columns_like_notebook(df_one, cols)
    df_one = _coerce_numeric(df_one)

    prob_model = float(model.predict_proba(df_one)[0, 1])

    shap_pack: Dict[str, Any] = {"ok": False}
    if ENABLE_SHAP:
        try:
            shap_pack = _compute_shap_explanation(model, df_one, sample_patient)
        except Exception:
            shap_pack = {"ok": False, "note": "SHAP computation error."}

    asm = sample_patient.get("current_asm", None)
    if use_rules and asm is not None and str(asm).strip() != "":
        prob_final, penalty, rule_reasons = apply_asm_rules(sample_patient, str(asm), prob_model)
    else:
        prob_final, penalty, rule_reasons = prob_model, 0.0, []

    pred_label = int(prob_final >= threshold)
    label_map = {0: "Not seizure-free at 12 months", 1: "Seizure-free at 12 months"}
    pretty = label_map.get(pred_label, str(pred_label))

    flags = _reliability_flags(sample_patient)
    applicability = _applicability_indicator(sample_patient)

    clinician_text = _clinician_summary(
        pretty_label=pretty,
        prob_final=prob_final,
        threshold=threshold,
        patient=sample_patient,
        reliability_flags=flags,
        asm_notes=rule_reasons,
        shap_doctor=(shap_pack.get("doctor") if shap_pack.get("ok") else None),
        shap_note=shap_pack.get("note", ""),
    )

    return {
        "pred_label": pred_label,
        "result_text": pretty,
        "prob_final": float(prob_final),
        "clinician_summary": clinician_text,
        "applicability_indicator": int(applicability),

        "shap": shap_pack,

        "ml_details": {
            "model_name": artifacts.get("model_name", "(unknown)"),
            "prob_model": float(prob_model),
            "prob_final": float(prob_final),
            "threshold": float(threshold),
            "rule_penalty": float(penalty),
            "rule_reasons_raw": rule_reasons,
            "reliability_flags": flags,
            "versions": artifacts.get("versions"),
        },
    }


def predict(sample_patient: Dict[str, Any], artifacts: Dict[str, Any]) -> Dict[str, Any]:
    return predict_patient(sample_patient, artifacts, use_rules=True)

