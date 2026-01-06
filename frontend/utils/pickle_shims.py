# frontend/utils/pickle_shims.py
from __future__ import annotations

import sys
import types
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


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
    """
    MUST match notebook:
    For ALL object/category cols:
      - lowercase, strip
      - "nan"/"null" -> np.nan
      - normalize whitespace
      - spaces -> underscores
    """
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
            "cns_infection", "substance_alcohol_abuse", "family_history"
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

        # 1) Duration since onset
        if {"age", "age_of_onset"}.issubset(X.columns):
            a = pd.to_numeric(X["age"], errors="coerce")
            o = pd.to_numeric(X["age_of_onset"], errors="coerce")
            X["duration_since_onset"] = (a - o).clip(lower=0)

        # 2) Seizure burden: log + high burden flag
        if "pretreatment_seizure_count" in X.columns:
            sc = pd.to_numeric(X["pretreatment_seizure_count"], errors="coerce")
            X["seizure_burden_log"] = np.log1p(sc)
            X["high_seizure_burden"] = (sc > self.med_seizure_).astype(int)

        # 3) Seizure frequency risk
        if {"pretreatment_seizure_count", "age"}.issubset(X.columns):
            sc = pd.to_numeric(X["pretreatment_seizure_count"], errors="coerce")
            a = pd.to_numeric(X["age"], errors="coerce")
            X["seizure_frequency_risk"] = sc / (a + 1)

        # 4) ASM exposure features
        if "prior_asm_exposure_count" in X.columns:
            p = pd.to_numeric(X["prior_asm_exposure_count"], errors="coerce")
            X["high_prior_asm"] = (p > 1).astype(int)
            X["poly_asm_history"] = (p >= 3).astype(int)

        # 5) Age-of-onset group
        if "age_of_onset" in X.columns:
            o = pd.to_numeric(X["age_of_onset"], errors="coerce")
            X["age_of_onset_group"] = pd.cut(
                o,
                bins=[-0.01, 1, 18, 40, 65, 120],
                labels=["Infant", "Childhood", "Young_Adult", "Mid_Age", "Older"],
                include_lowest=True
            ).astype(str)

        # 6) Structural lesion flag from MRI (no premature fill)
        if "mri_lesion_type" in X.columns:
            s = X["mri_lesion_type"].astype(str).str.strip().str.lower()
            s = s.replace({"nan": np.nan, "null": np.nan})
            X["structural_lesion_flag"] = (s.notna() & (s != "none") & (s != "normal")).astype(int)

        # 7) EEG epileptic flag
        if "eeg_status_detail" in X.columns:
            s = X["eeg_status_detail"].astype(str).str.strip().str.lower()
            s = s.replace({"nan": np.nan, "null": np.nan})
            X["eeg_epileptic_flag"] = (s.notna() & (s != "normal")).astype(int)

        # 8) Comorbidity burden
        existing = [c for c in self.comorb_cols if c in X.columns]
        if existing:
            for c in existing:
                X[c + "_bin"] = X[c].apply(self.norm_yes_no)
            bin_cols = [c + "_bin" for c in existing]
            X["comorbidity_burden"] = X[bin_cols].sum(axis=1)
            X["high_comorbidity"] = (X["comorbidity_burden"] > 0).astype(int)

        # 9) Clinical risk index (engineered)
        risk_components = [
            c for c in [
                "high_seizure_burden",
                "structural_lesion_flag",
                "high_prior_asm",
                "high_comorbidity",
                "eeg_epileptic_flag",
            ]
            if c in X.columns
        ]
        if risk_components:
            X["clinical_risk_index"] = X[risk_components].sum(axis=1)

        return X


def _register_fake_module(module_name: str) -> types.ModuleType:
    if module_name not in sys.modules:
        sys.modules[module_name] = types.ModuleType(module_name)
    return sys.modules[module_name]


def register_pickle_shims() -> None:
    for mod_name in ("__main__", "main"):
        mod = _register_fake_module(mod_name)
        setattr(mod, "ColumnNameCleaner", ColumnNameCleaner)
        setattr(mod, "CategoricalValueCleaner", CategoricalValueCleaner)
        setattr(mod, "FeatureEngineer", FeatureEngineer)
