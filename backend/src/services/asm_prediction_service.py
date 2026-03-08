# backend/src/services/asm_prediction_service.py
from __future__ import annotations

import sys
import types
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.50
ENABLE_SHAP = True
SHAP_TOP_N = 6
SHAP_MAX_FEATURES_IN_TEXT = 5

# Raw categorical columns expected by the model
_CAT_BASE_COLS = {
    "sex", "seizure_type", "current_asm", "mri_lesion_type",
    "eeg_status_detail", "psychiatric_disorder", "intellectual_disability",
    "cerebrovascular_disease", "head_trauma", "cns_infection",
    "substance_alcohol_abuse", "family_history",
}

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
# Pickle shim classes (must match training notebook exactly)
# ============================================================
class ColumnNameCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X.columns = (
            X.columns.astype(str)
            .str.strip().str.lower()
            .str.replace(" ", "_", regex=False)
        )
        return X


class CategoricalValueCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        obj_cols = X.select_dtypes(include=["object", "category"]).columns
        for c in obj_cols:
            s = X[c].astype(str).str.strip().str.lower()
            s = s.replace({"nan": np.nan, "null": np.nan})
            s = s.str.replace(r"\s+", " ", regex=True).str.strip()
            s = s.str.replace(" ", "_", regex=False)
            X[c] = s
        return X


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.comorb_cols = [
            "psychiatric_disorder", "intellectual_disability",
            "cerebrovascular_disease", "head_trauma",
            "cns_infection", "substance_alcohol_abuse", "family_history",
        ]

    @staticmethod
    def norm_yes_no(v):
        v = str(v).strip().lower()
        return 1 if v in ["yes", "y", "true", "1"] else 0

    def fit(self, X, y=None):
        X = X.copy()
        if "pretreatment_seizure_count" in X.columns:
            med = pd.to_numeric(X["pretreatment_seizure_count"], errors="coerce").median()
            self.med_seizure_ = 0.0 if pd.isna(med) else float(med)
        else:
            self.med_seizure_ = 0.0
        return self

    def transform(self, X):
        X = X.copy()

        if {"age", "age_of_onset"}.issubset(X.columns):
            a = pd.to_numeric(X["age"], errors="coerce")
            o = pd.to_numeric(X["age_of_onset"], errors="coerce")
            X["duration_since_onset"] = (a - o).clip(lower=0)

        if "pretreatment_seizure_count" in X.columns:
            sc = pd.to_numeric(X["pretreatment_seizure_count"], errors="coerce")
            X["seizure_burden_log"] = np.log1p(sc)
            X["high_seizure_burden"] = (sc > self.med_seizure_).astype(int)

        if {"pretreatment_seizure_count", "age"}.issubset(X.columns):
            sc = pd.to_numeric(X["pretreatment_seizure_count"], errors="coerce")
            a = pd.to_numeric(X["age"], errors="coerce")
            X["seizure_frequency_risk"] = sc / (a + 1)

        if "prior_asm_exposure_count" in X.columns:
            p = pd.to_numeric(X["prior_asm_exposure_count"], errors="coerce")
            X["high_prior_asm"] = (p > 1).astype(int)
            X["poly_asm_history"] = (p >= 3).astype(int)

        if "age_of_onset" in X.columns:
            o = pd.to_numeric(X["age_of_onset"], errors="coerce")
            X["age_of_onset_group"] = pd.cut(
                o,
                bins=[-0.01, 1, 18, 40, 65, 120],
                labels=["Infant", "Childhood", "Young_Adult", "Mid_Age", "Older"],
                include_lowest=True,
            ).astype(str)

        if "mri_lesion_type" in X.columns:
            s = X["mri_lesion_type"].astype(str).str.strip().str.lower()
            s = s.replace({"nan": np.nan, "null": np.nan})
            X["structural_lesion_flag"] = (s.notna() & (s != "none") & (s != "normal")).astype(int)

        if "eeg_status_detail" in X.columns:
            s = X["eeg_status_detail"].astype(str).str.strip().str.lower()
            s = s.replace({"nan": np.nan, "null": np.nan})
            X["eeg_epileptic_flag"] = (s.notna() & (s != "normal")).astype(int)

        existing = [c for c in self.comorb_cols if c in X.columns]
        if existing:
            for c in existing:
                X[c + "_bin"] = X[c].apply(self.norm_yes_no)
            bin_cols = [c + "_bin" for c in existing]
            X["comorbidity_burden"] = X[bin_cols].sum(axis=1)
            X["high_comorbidity"] = (X["comorbidity_burden"] > 0).astype(int)

        risk_components = [
            c for c in [
                "high_seizure_burden", "structural_lesion_flag",
                "high_prior_asm", "high_comorbidity", "eeg_epileptic_flag",
            ]
            if c in X.columns
        ]
        if risk_components:
            X["clinical_risk_index"] = X[risk_components].sum(axis=1)

        return X


def _register_pickle_shims() -> None:
    """Register shim classes so joblib can unpickle the trained model."""
    for mod_name in ("__main__", "main"):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)
        mod = sys.modules[mod_name]
        setattr(mod, "ColumnNameCleaner", ColumnNameCleaner)
        setattr(mod, "CategoricalValueCleaner", CategoricalValueCleaner)
        setattr(mod, "FeatureEngineer", FeatureEngineer)


# ============================================================
# ASM clinical rule layer
# ============================================================
def _norm_str(v: Any) -> str:
    return "" if v is None else str(v).strip().lower()


def _norm_yes(v: Any) -> bool:
    return _norm_str(v) in {"yes", "y", "true", "1"}


def apply_asm_rules(
    patient: Dict[str, Any], asm: str, base_prob: float
) -> Tuple[float, float, List[str]]:
    asm = _norm_str(asm)
    sex = _norm_str(patient.get("sex"))
    seizure_type = _norm_str(patient.get("seizure_type"))

    age = patient.get("age")
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
        if sex == "female" and age is not None and age < 50:
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
# SHAP explanation helpers
# ============================================================
def _extract_fitted_pipeline_and_estimator(model: Any) -> Tuple[Optional[Any], Optional[Any]]:
    if hasattr(model, "named_steps"):
        pipe = model
        est = getattr(pipe, "steps", [])[-1][1] if getattr(pipe, "steps", None) else None
        return pipe, est

    try:
        cc0 = model.calibrated_classifiers_[0]
        base = getattr(cc0, "estimator", None) or getattr(cc0, "base_estimator", None)
        if base is not None:
            if hasattr(base, "named_steps"):
                return base, getattr(base, "steps", [])[-1][1]
            return None, base
    except Exception:
        pass

    return None, None


def _transform_to_model_matrix(pipe: Any, raw_df: pd.DataFrame) -> Tuple[np.ndarray, Optional[List[str]]]:
    X = raw_df.copy()
    if hasattr(pipe, "named_steps"):
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
                pass
            return Xt, names
    return X.to_numpy(), list(X.columns)


def _pretty_feature_name(name: str) -> str:
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
            return f"Epilepsy duration (approx. {max(age - onset, 0):.0f} years)"
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


def _summarize_shap_for_doctor(
    shap_rows: List[Dict[str, Any]], patient: Dict[str, Any]
) -> Dict[str, Any]:
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


def _compute_shap_explanation(
    model: Any, df_one: pd.DataFrame, patient: Dict[str, Any]
) -> Dict[str, Any]:
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
            explainer = shap.KernelExplainer(est.predict_proba, Xt_dense)
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
        rows.append({
            "feature": _pretty_feature_name(feature_names[i]),
            "shap": val,
            "direction": "supports" if val > 0 else "against",
        })

    doctor = _summarize_shap_for_doctor(rows, patient=patient)
    return {"ok": True, "top": rows, "doctor": doctor}


# ============================================================
# Reliability / applicability guards
# ============================================================
def _reliability_flags(x: Dict[str, Any]) -> List[str]:
    flags: List[str] = []

    def _f(v):
        try:
            return float(v)
        except Exception:
            return None

    age = _f(x.get("age"))
    onset = _f(x.get("age_of_onset"))
    ptsc = _f(x.get("pretreatment_seizure_count"))
    prior = _f(x.get("prior_asm_exposure_count"))

    if age is not None and (age < 0 or age > 120):
        flags.append("Age outside plausible range (0-120).")
    if age is not None and onset is not None and onset > age:
        flags.append("Age of onset greater than current age (inconsistent).")
    if ptsc is not None and ptsc > 500:
        flags.append("Pre-treatment seizure count extremely high; probability may be less reliable.")
    if prior is not None and prior > 50:
        flags.append("Prior ASM exposure unusually high; probability may be less reliable.")
    return flags


def _applicability_indicator(x: Dict[str, Any]) -> int:
    score = 0

    def _f(v):
        try:
            return float(v)
        except Exception:
            return None

    ptsc = _f(x.get("pretreatment_seizure_count"))
    if ptsc is not None:
        score += 2 if ptsc >= 20 else (1 if ptsc >= 10 else 0)

    prior = _f(x.get("prior_asm_exposure_count"))
    if prior is not None:
        score += 2 if prior >= 3 else (1 if prior >= 1 else 0)

    eeg = (x.get("eeg_status_detail") or "").strip().lower()
    if eeg in {"focal", "generalized", "multifocal"}:
        score += 1

    mri = x.get("mri_lesion_type")
    if mri and str(mri).strip().lower() not in {"select", "select an option", ""}:
        score += 1

    return int(min(5, max(0, score)))


# ============================================================
# Clinician narrative
# ============================================================
def _prob_band(prob: float) -> str:
    if prob >= 0.80:
        return "High"
    if prob >= 0.60:
        return "Moderate-high"
    if prob >= 0.40:
        return "Intermediate"
    return "Low"


def _format_percent(prob: float) -> str:
    return f"{float(prob) * 100:.0f}%"


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

    lines: List[str] = [impression, ""]

    if shap_doctor and (supports or against):
        if supports:
            lines.append("**Factors that increased the estimate:**")
            for x in supports:
                lines.append(f"- {x}")
        if against:
            lines.append("**Factors that reduced the estimate:**")
            for x in against:
                lines.append(f"- {x}")
        if shap_note:
            lines += ["", f"*Note:* {shap_note}"]
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
    lines.append("- Reassess at follow-up with early response (first 4-12 weeks often changes prognosis).")
    lines.append("")

    return "\n".join(lines)


# ============================================================
# Main service class
# ============================================================
class ASMPredictionService:
    """Loads the trained ASM response prediction model and serves predictions."""

    def __init__(self, model_path: str | Path):
        _register_pickle_shims()

        import joblib

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"ASM model artifact not found: {path.resolve()}")

        obj = joblib.load(path)
        if not isinstance(obj, dict):
            obj = {"model": obj}

        self.model = obj.get("model")
        if self.model is None or not hasattr(self.model, "predict_proba"):
            raise TypeError("Loaded ASM artifact does not contain a valid model with predict_proba().")

        self.threshold = float(obj.get("threshold", DEFAULT_THRESHOLD))
        self.feature_columns_raw: Optional[List[str]] = obj.get("feature_columns_raw")
        self.model_name: str = obj.get("best_base_model") or obj.get("model_name") or type(self.model).__name__
        self.available_asms: List[str] = obj.get("available_asms", [])

        logger.info(
            f"ASM model loaded: {self.model_name}, "
            f"threshold={self.threshold:.3f}, "
            f"features={len(self.feature_columns_raw) if self.feature_columns_raw else '?'}"
        )

    def _align_columns(self, df_one: pd.DataFrame) -> pd.DataFrame:
        if not self.feature_columns_raw:
            return df_one
        df_one = df_one.copy()
        for c in self.feature_columns_raw:
            if c not in df_one.columns:
                df_one[c] = "unknown" if c in _CAT_BASE_COLS else 0
        return df_one[self.feature_columns_raw].copy()

    @staticmethod
    def _coerce_numeric(df_one: pd.DataFrame) -> pd.DataFrame:
        df_one = df_one.copy()
        for c in ["age", "age_of_onset", "pretreatment_seizure_count", "prior_asm_exposure_count"]:
            if c in df_one.columns:
                df_one[c] = pd.to_numeric(df_one[c], errors="coerce")
        return df_one

    def predict(self, sample_patient: Dict[str, Any]) -> Dict[str, Any]:
        """Run the full prediction pipeline for a single patient dict."""
        # Build a one-row DataFrame
        df_one = pd.DataFrame([{k: (np.nan if v is None else v) for k, v in sample_patient.items()}])
        df_one = self._align_columns(df_one)
        df_one = self._coerce_numeric(df_one)

        # Base model probability
        prob_model = float(self.model.predict_proba(df_one)[0, 1])

        # SHAP explanation
        shap_pack: Dict[str, Any] = {"ok": False}
        if ENABLE_SHAP:
            try:
                shap_pack = _compute_shap_explanation(self.model, df_one, sample_patient)
            except Exception:
                shap_pack = {"ok": False, "note": "SHAP computation error."}

        # ASM clinical rule adjustment
        asm = sample_patient.get("current_asm")
        if asm is not None and str(asm).strip() != "":
            prob_final, penalty, rule_reasons = apply_asm_rules(sample_patient, str(asm), prob_model)
        else:
            prob_final, penalty, rule_reasons = prob_model, 0.0, []

        pred_label = int(prob_final >= self.threshold)
        label_map = {0: "Not seizure-free at 12 months", 1: "Seizure-free at 12 months"}
        pretty = label_map.get(pred_label, str(pred_label))

        flags = _reliability_flags(sample_patient)
        applicability = _applicability_indicator(sample_patient)

        clinician_text = _clinician_summary(
            pretty_label=pretty,
            prob_final=prob_final,
            threshold=self.threshold,
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
                "model_name": self.model_name,
                "prob_model": float(prob_model),
                "prob_final": float(prob_final),
                "threshold": float(self.threshold),
                "rule_penalty": float(penalty),
                "rule_reasons_raw": rule_reasons,
                "reliability_flags": flags,
            },
        }
