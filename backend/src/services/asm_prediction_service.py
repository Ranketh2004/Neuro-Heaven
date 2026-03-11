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

# Clinical profile metadata for each ASM (evidence-based, ILAE 2022 guidelines)
_ASM_CLINICAL_PROFILE: Dict[str, Dict[str, str]] = {
    "levetiracetam": {
        "tier": "First-line",
        "spectrum": "Broad (Focal + Generalized)",
        "mechanism": "SV2A synaptic vesicle modulator",
        "teratogenic_risk": "Low",
        "monitoring": "Mood/behaviour; renal dose adjustment",
    },
    "lamotrigine": {
        "tier": "First-line",
        "spectrum": "Broad (Focal + Generalized)",
        "mechanism": "Voltage-gated Na⁺ channel blocker",
        "teratogenic_risk": "Low–Moderate",
        "monitoring": "Titration schedule; rash (SJS risk); drug interactions",
    },
    "valproate": {
        "tier": "First-line (Generalised epilepsy)",
        "spectrum": "Broad (Generalised > Focal)",
        "mechanism": "Na⁺ channel / GABA / T-Ca²⁺ modulator",
        "teratogenic_risk": "High (VALPROATE PREVENT program compliance required)",
        "monitoring": "LFTs, weight, ammonia, teratogenicity counselling",
    },
    "carbamazepine": {
        "tier": "First-line (Focal epilepsy only)",
        "spectrum": "Focal only — avoid in generalised",
        "mechanism": "Voltage-gated Na⁺ channel blocker",
        "teratogenic_risk": "Moderate",
        "monitoring": "Na⁺ levels (hyponatraemia), CBC, LFTs, drug interactions (enzyme inducer)",
    },
    "phenobarbital": {
        "tier": "Third-line (or resource-limited settings)",
        "spectrum": "Broad",
        "mechanism": "GABA-A potentiator",
        "teratogenic_risk": "Moderate",
        "monitoring": "Sedation, cognition, dependence; enzyme inducer",
    },
    "phenytoin": {
        "tier": "Second/Third-line (narrow therapeutic index)",
        "spectrum": "Focal ± Generalised tonic-clonic",
        "mechanism": "Voltage-gated Na⁺ channel blocker",
        "teratogenic_risk": "Moderate",
        "monitoring": "Drug levels, gingival hyperplasia, ataxia, cardiac (if IV), enzyme inducer",
    },
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
) -> Tuple[float, float, List[str], List[str]]:
    """Return (adjusted_prob, penalty, caution_notes, benefit_notes)."""
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
    febrile = _norm_yes(patient.get("febrile_convulsion"))
    fam_hx = _norm_yes(patient.get("family_history"))

    try:
        sc = float(patient.get("pretreatment_seizure_count", 0) or 0)
    except Exception:
        sc = 0.0

    is_focal = seizure_type in {"focal", "focal_onset", "partial"}
    is_generalised = seizure_type in {"generalized", "generalised", "absence", "myoclonic", "tonic-clonic", "tonic_clonic", "jme"}
    is_myoclonic = seizure_type in {"myoclonic", "jme"}
    is_wocbp = (sex == "female" and age is not None and age < 50)

    penalty = 0.0
    caution_notes: List[str] = []
    benefit_notes: List[str] = []

    # ------------------------------------------------------------------
    # Levetiracetam
    # ------------------------------------------------------------------
    if asm == "levetiracetam":
        # Benefits
        if is_focal:
            penalty -= 0.06
            benefit_notes.append("Preferred broad-spectrum option for focal onset seizures (ILAE Grade A).")
        if is_generalised and not is_myoclonic:
            penalty -= 0.04
            benefit_notes.append("Broad-spectrum; suitable for generalised tonic-clonic seizures — ILAE-endorsed option for generalised epilepsy.")
        if is_myoclonic:
            penalty -= 0.03
            benefit_notes.append("Effective adjunct for myoclonic seizures in JME; first alternative when valproate is contraindicated.")
        if is_wocbp:
            penalty -= 0.07
            benefit_notes.append("Preferred in women of childbearing potential — no significant teratogenic risk.")
        if head_trauma or cns_inf:
            penalty -= 0.04
            benefit_notes.append("Broad-spectrum; frequently used in structural/acquired epilepsy (trauma, CNS infection).")
        if febrile:
            penalty -= 0.02
            benefit_notes.append("Reasonable option in patients with history of febrile convulsions.")
        # Cautions
        if psych:
            penalty += 0.12
            caution_notes.append("Levetiracetam: psychiatric/mood side-effects (irritability, depression) — monitor closely.")
        if alcohol:
            penalty += 0.04
            caution_notes.append("Levetiracetam: alcohol/substance use increases behavioural risk and reduces adherence.")

    # ------------------------------------------------------------------
    # Lamotrigine
    # ------------------------------------------------------------------
    elif asm == "lamotrigine":
        # Benefits
        if is_focal:
            penalty -= 0.04
            benefit_notes.append("Effective for focal onset seizures; well-tolerated long-term.")
        if seizure_type == "absence":
            penalty -= 0.06
            benefit_notes.append("Effective for childhood and juvenile absence epilepsy; second-line after valproate per ILAE 2022.")
        elif is_generalised and not is_myoclonic:
            penalty -= 0.03
            benefit_notes.append("Reasonable for generalised tonic-clonic seizures as adjunct or monotherapy; second-line option.")
        if is_wocbp or sex == "female":
            penalty -= 0.05
            benefit_notes.append("Preferred in women of childbearing potential — comparatively lower teratogenic risk.")
        if psych:
            penalty -= 0.05
            benefit_notes.append("Mood-stabilising properties; may benefit patients with comorbid mood disorders.")
        # Cautions
        if is_myoclonic:
            penalty += 0.12
            caution_notes.append("Lamotrigine: may worsen myoclonic seizures (JME) — avoid or use with caution.")
        if sc >= 30:
            penalty += 0.05
            caution_notes.append("Lamotrigine: mandatory slow titration may delay adequate seizure control in high-burden cases.")
        if sc >= 20 and not is_myoclonic:
            penalty += 0.02
            caution_notes.append("Lamotrigine: titration schedule should be carefully observed; interaction with valproate alters levels.")

    # ------------------------------------------------------------------
    # Valproate
    # ------------------------------------------------------------------
    elif asm == "valproate":
        # Benefits — is_myoclonic must be checked first; it is a subset of is_generalised
        if is_myoclonic:
            penalty -= 0.12
            benefit_notes.append("Drug of choice for myoclonic seizures and JME — superior efficacy over LTG/LEV for myoclonic and absence components (ILAE Grade A).")
        elif is_generalised:
            penalty -= 0.10
            benefit_notes.append("First-choice for generalised epilepsy syndromes (absence, generalised tonic-clonic) — broadest spectrum evidence base (ILAE Grade A).")
        if fam_hx:
            penalty -= 0.02
            benefit_notes.append("Broad-spectrum coverage useful in idiopathic generalised epilepsy with family history.")
        # Cautions
        if is_wocbp:
            penalty += 0.25
            caution_notes.append("Valproate: HIGH teratogenic risk in females of childbearing potential (neural tube defects, neurodevelopmental effects) — AVOID unless no alternatives; comply with VALPROATE PREVENT programme.")
        if alcohol:
            penalty += 0.05
            caution_notes.append("Valproate: alcohol use complicates hepatic monitoring and adherence (hepatotoxicity risk).")
        if int_dis:
            penalty += 0.04
            caution_notes.append("Valproate: cognitive/sedation burden may be additive — monitor closely in intellectual disability.")

    # ------------------------------------------------------------------
    # Carbamazepine
    # ------------------------------------------------------------------
    elif asm == "carbamazepine":
        # Benefits
        if is_focal:
            penalty -= 0.08
            benefit_notes.append("First-line for focal onset epilepsy; well-established long-term data.")
        # Cautions
        if is_generalised:
            penalty += 0.20
            caution_notes.append("Carbamazepine: CONTRAINDICATED in generalised epilepsy — may aggravate absence, myoclonic, and atonic seizures.")
        if seizure_type == "mixed":
            penalty += 0.12
            caution_notes.append("Carbamazepine: caution in mixed seizure disorder — risk of aggravating any generalised (absence, myoclonic) components; consider a broad-spectrum agent.")
        if age is not None and age >= 65:
            penalty += 0.07
            caution_notes.append("Carbamazepine: elderly — higher risk of hyponatraemia, dizziness, falls, and drug interactions.")
        if cerebro:
            penalty += 0.06
            caution_notes.append("Carbamazepine: caution with cerebrovascular disease — tolerability and interaction profile.")
        if alcohol:
            penalty += 0.05
            caution_notes.append("Carbamazepine: enzyme induction alters drug levels; adherence concerns with alcohol use.")

    # ------------------------------------------------------------------
    # Phenobarbital
    # ------------------------------------------------------------------
    elif asm == "phenobarbital":
        # Benefits — limited; used in resource-limited/refractory settings
        if head_trauma or cns_inf:
            penalty -= 0.01
            benefit_notes.append("Phenobarbital: historically used in acute structural epilepsy contexts; broad-spectrum.")
        # Cautions
        if int_dis:
            penalty += 0.18
            caution_notes.append("Phenobarbital: significant sedation and cognitive impairment — strongly consider alternatives in intellectual disability.")
        if psych:
            penalty += 0.10
            caution_notes.append("Phenobarbital: may worsen depression and behavioural symptoms — avoid in psychiatric comorbidity where possible.")
        if age is not None and age >= 65:
            penalty += 0.18
            caution_notes.append("Phenobarbital: elderly — high sedation, falls risk, paradoxical agitation; generally avoided.")
        if cerebro:
            penalty += 0.10
            caution_notes.append("Phenobarbital: sedation and falls risk particularly elevated with cerebrovascular disease.")
        if alcohol:
            penalty += 0.10
            caution_notes.append("Phenobarbital: combined CNS/respiratory depression with alcohol — avoid.")
        if not (int_dis or psych or (age is not None and age >= 65) or cerebro or alcohol):
            caution_notes.append("Phenobarbital: third-line agent — consider only after LEV/LTG/VPA failure; enzyme-inducing, sedating.")

    # ------------------------------------------------------------------
    # Phenytoin
    # ------------------------------------------------------------------
    elif asm == "phenytoin":
        # Benefits — mainly acute/IV use
        if is_focal and not (age is not None and age >= 65) and not cerebro:
            benefit_notes.append("Phenytoin: option for focal epilepsy in the absence of tolerability contraindications.")
        # Cautions
        if is_myoclonic or seizure_type == "absence":
            penalty += 0.22
            caution_notes.append("Phenytoin: CONTRAINDICATED — known to aggravate absence and myoclonic seizures; use a broad-spectrum agent (valproate, lamotrigine, levetiracetam).")
        if age is not None and age >= 65:
            penalty += 0.13
            caution_notes.append("Phenytoin: elderly — narrow therapeutic index, high inter-individual PK variability, ataxia, falls risk.")
        if cerebro:
            penalty += 0.09
            caution_notes.append("Phenytoin: cardiac conduction effects and drug interactions are heightened in cerebrovascular disease.")
        if alcohol:
            penalty += 0.07
            caution_notes.append("Phenytoin: alcohol causes unpredictable level fluctuations and impairs adherence.")
        if int_dis:
            penalty += 0.07
            caution_notes.append("Phenytoin: cognitive side-effects and cosmetic effects (gingival hyperplasia) are poorly tolerated.")
        if not caution_notes and not benefit_notes:
            caution_notes.append("Phenytoin: narrow therapeutic index and non-linear kinetics require therapeutic drug monitoring.")

    # Floor penalty so a cascade of benefits does not go below −0.10
    penalty = max(penalty, -0.10)
    adjusted = float(np.clip(base_prob - penalty, 0.0, 0.95))
    return adjusted, float(penalty), caution_notes, benefit_notes


def _suitability_class(penalty: float) -> str:
    """Classify a net penalty into a clinical suitability tier."""
    if penalty < -0.02:
        return "preferred"
    if penalty <= 0.05:
        return "acceptable"
    if penalty <= 0.15:
        return "caution"
    return "avoid"


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
        try:
            n = int(float(sc)) if sc else 0
            context = "low burden — favours better response" if n <= 3 else (
                "moderate burden" if n <= 10 else "high burden — associated with reduced likelihood of seizure freedom"
            )
        except Exception:
            context = ""
        return f"Pre-treatment seizure count (n={sc}; {context})" if context else f"Pre-treatment seizure count (n={sc})"

    if "number of prior asms" in fl or "prior asms" in fl or "prior_asm" in fl:
        pa = _value_str(patient, "prior_asm_exposure_count")
        try:
            n = int(float(pa)) if pa else 0
            context = "no prior treatment — favourable prognostic indicator" if n == 0 else (
                f"{n} prior ASM(s) — each failed ASM reduces probability of subsequent remission"
            )
        except Exception:
            context = ""
        return f"Prior ASM exposure ({context})" if context else f"Prior anti-seizure medications tried (n={pa})"

    if "age at seizure onset" in fl:
        ao = _value_str(patient, "age_of_onset")
        try:
            age_val = float(ao) if ao else None
            if age_val is not None:
                if age_val < 2:
                    context = "neonatal/infantile onset — associated with complex underlying aetiology"
                elif age_val < 12:
                    context = "childhood onset — prognosis varies widely by syndrome"
                elif age_val < 18:
                    context = "adolescent onset — consider idiopathic generalised syndromes"
                else:
                    context = "adult onset — structural/metabolic aetiology more likely"
            else:
                context = ""
        except Exception:
            context = ""
        return f"Age at seizure onset ({ao} yrs; {context})" if context else f"Age at seizure onset ({ao} yrs)"

    if fl == "current age" or "current age" in fl:
        age = _value_str(patient, "age")
        return f"Current patient age ({age} yrs)"

    if "duration since onset" in fl:
        try:
            age = float(patient.get("age"))
            onset = float(patient.get("age_of_onset"))
            dur = max(age - onset, 0)
            context = "recent onset — early treatment response is a strong outcome predictor" if dur <= 2 else (
                "long-standing epilepsy — chronic course may reflect treatment-refractory disease" if dur > 10 else ""
            )
            return f"Epilepsy duration ({dur:.0f} yrs; {context})" if context else f"Epilepsy duration ({dur:.0f} yrs)"
        except Exception:
            return "Epilepsy duration"

    if "seizure frequency relative to age" in fl or "seizure_frequency_risk" in fl:
        return "Seizure frequency relative to age (high frequency/age ratio worsens prognosis)"

    if "overall seizure burden" in fl or "seizure_burden_log" in fl:
        return "Overall seizure burden score (greater burden reduces likelihood of pharmacological remission)"

    if "comorbidity" in fl:
        return "Comorbidity burden (psychiatric, cognitive, or medical comorbidities reduce seizure-freedom probability)"

    if "clinical_risk_index" in fl or "clinical risk index" in fl:
        return "Clinical risk index (composite of structural lesion, EEG, seizure burden, comorbidity, and prior treatment)"

    if "high_prior_asm" in fl or "more than one prior" in fl:
        return "History of ≥2 prior ASMs — significantly associated with drug-resistant epilepsy"

    if "poly_asm" in fl or "three or more prior" in fl:
        return "≥3 prior ASMs (drug-resistant epilepsy criteria met — consider tertiary evaluation)"

    if "structural_lesion" in fl or "structural lesion" in fl:
        return "Structural MRI lesion (structural aetiology reduces probability of pharmacological remission)"

    if "eeg_epileptic" in fl or ("eeg" in fl and "flag" in fl):
        return "Epileptiform EEG activity (ictal/interictal abnormality correlates with active epilepsy)"

    if "eeg" in fl:
        eeg = _value_str(patient, "eeg_status_detail")
        eeg_map = {
            "focal": "focal interictal discharges — supports focal epilepsy diagnosis",
            "generalized": "generalised discharges — supports idiopathic generalised epilepsy",
            "multifocal": "multifocal discharges — may indicate diffuse/structural aetiology",
            "normal": "normal EEG — may reduce diagnostic certainty",
        }
        eeg_clean = str(eeg or "").strip().lower()
        context = eeg_map.get(eeg_clean, "")
        return f"EEG findings ({eeg}; {context})" if context else f"EEG findings ({eeg})"

    if "mri lesion type" in fl or "structural lesion" in fl:
        mri = _value_str(patient, "mri_lesion_type")
        if mri and str(mri).strip().lower() not in {"select", "select an option"}:
            mri_map = {
                "hippocampal_sclerosis": "hippocampal sclerosis — surgically remediable; pharmacological remission less likely",
                "tumor": "tumour-related epilepsy — seizure freedom depends on extent of surgical resection",
                "cortical_dysplasia": "focal cortical dysplasia — often treatment-resistant; surgical evaluation warranted",
            }
            context = mri_map.get(str(mri).strip().lower(), "")
            return f"MRI lesion ({mri}; {context})" if context else f"MRI lesion type ({mri})"
        return "MRI lesion information"

    if "mri lesion type:" in feature_label:
        return feature_label.replace("MRI lesion type:", "MRI finding:")

    if "seizure type:" in feature_label:
        st_map = {
            "Seizure type: Focal": "Seizure type: Focal (focal onset — aetiology drives prognosis)",
            "Seizure type: Generalized": "Seizure type: Generalised (idiopathic generalised epilepsy syndromes often respond well to broad-spectrum agents)",
            "Seizure type: Myoclonic": "Seizure type: Myoclonic (JME — typically controlled with VPA/LEV/LTG; lifelong treatment often required)",
        }
        return st_map.get(feature_label, feature_label)

    if "sex" in fl:
        sex = _value_str(patient, "sex")
        return f"Biological sex ({sex}) — influences ASM tolerability and teratogenicity considerations"

    if "current asm" in fl:
        asm = _value_str(patient, "current_asm")
        return f"Current ASM ({asm}) — efficacy and tolerability profile shapes overall outcome estimate"

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
            f"This patient's clinical profile is associated with a **{pct} estimated probability of "
            f"seizure freedom at 12 months** ({band} likelihood). The model output favours a positive "
            f"treatment response; however, this estimate reflects population-level associations and "
            f"must be interpreted alongside individual clinical context, comorbidity burden, and "
            f"medication adherence."
        )
    else:
        impression = (
            f"This patient's clinical profile is associated with a **{pct} estimated probability of "
            f"seizure freedom at 12 months** ({band} likelihood). The model output suggests suboptimal "
            f"seizure control is likely without treatment modification. Consider reviewing current ASM "
            f"selection, adherence, and whether further diagnostic workup (e.g., video-EEG, MRI) "
            f"or specialist referral is warranted."
        )

    supports, against = [], []
    if shap_doctor:
        supports = shap_doctor.get("supports", [])[:SHAP_MAX_FEATURES_IN_TEXT]
        against = shap_doctor.get("against", [])[:SHAP_MAX_FEATURES_IN_TEXT]

    lines: List[str] = [impression, ""]

    if shap_doctor and (supports or against):
        if supports:
            lines.append("**Clinical factors favouring seizure freedom:**")
            for x in supports:
                lines.append(f"- {x}")
        if against:
            lines.append("**Clinical factors reducing likelihood of seizure freedom:**")
            for x in against:
                lines.append(f"- {x}")
        if shap_note:
            lines += ["", f"*Note:* {shap_note}"]
        lines.append("")

    if asm_notes:
        lines.append("**Current ASM — clinical considerations:**")
        for n in asm_notes[:3]:
            lines.append(f"- {n}")
        lines.append("")

    if reliability_flags:
        lines.append("**Data quality / applicability flags:**")
        for f in reliability_flags[:2]:
            lines.append(f"- {f}")
        lines.append("")

    lines.append("**Recommended next steps:**")
    lines.append("- Verify seizure frequency data and confirm current medication adherence before acting on this estimate.")
    lines.append("- Ensure EEG and MRI findings have been formally reported and correlated with clinical semiology.")
    lines.append("- Reassess treatment response at 4–12 weeks; early seizure reduction is a strong independent predictor of 12-month outcome.")
    lines.append("- If two or more appropriately dosed ASMs have failed, consider referral to a tertiary epilepsy centre for comprehensive evaluation.")
    lines.append("- Document adverse effects, quality-of-life impact, and seizure severity at each clinical review, not seizure count alone.")
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
            prob_final, penalty, rule_cautions, rule_benefits = apply_asm_rules(sample_patient, str(asm), prob_model)
            rule_reasons = rule_cautions + rule_benefits
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

    def rank_asms(self, patient: Dict[str, Any]) -> Dict[str, Any]:
        """Rank all available ASMs by rule-adjusted seizure-freedom probability."""
        df_one = pd.DataFrame([{k: (np.nan if v is None else v) for k, v in patient.items()}])
        df_one = self._align_columns(df_one)
        df_one = self._coerce_numeric(df_one)

        # Base probability is the same for every ASM — reflects patient clinical profile.
        prob_model = float(self.model.predict_proba(df_one)[0, 1])

        asms_to_rank = self.available_asms if self.available_asms else [
            "Levetiracetam", "Lamotrigine", "Valproate",
            "Phenobarbital", "Carbamazepine", "Phenytoin",
        ]

        rankings: List[Dict[str, Any]] = []
        for asm in asms_to_rank:
            adjusted_prob, penalty, caution_notes, benefit_notes = apply_asm_rules(patient, asm, prob_model)
            profile = _ASM_CLINICAL_PROFILE.get(asm.lower(), {})
            rankings.append({
                "asm": asm,
                "prob_adjusted": float(adjusted_prob),
                "prob_base": float(prob_model),
                "penalty": float(penalty),
                "caution_notes": caution_notes,
                "benefit_notes": benefit_notes,
                "rule_notes": caution_notes,   # backward-compat alias
                "pred_label": int(adjusted_prob >= self.threshold),
                "tier": profile.get("tier", ""),
                "spectrum": profile.get("spectrum", ""),
                "teratogenic_risk": profile.get("teratogenic_risk", ""),
                "monitoring": profile.get("monitoring", ""),
                "suitability": _suitability_class(penalty),
            })

        rankings.sort(key=lambda x: x["prob_adjusted"], reverse=True)

        flags = _reliability_flags(patient)
        applicability = _applicability_indicator(patient)

        return {
            "rankings": rankings,
            "prob_base": float(prob_model),
            "reliability_flags": flags,
            "applicability_indicator": int(applicability),
        }
