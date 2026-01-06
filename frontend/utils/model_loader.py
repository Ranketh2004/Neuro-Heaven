# utils/pickle_shims.py
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
            .str.strip()
            .str.lower()
            .str.replace(" ", "_", regex=False)
        )
        return X


class CategoricalValueCleaner(BaseEstimator, TransformerMixin):
    """
    Backward-compatible:
    - Handles old pickles where `unknown_token` was not present.
    """

    def __init__(self, unknown_token: str = "Unknown"):
        self.unknown_token = unknown_token

    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {})
        if "unknown_token" not in self.__dict__ or self.__dict__["unknown_token"] is None:
            self.__dict__["unknown_token"] = "Unknown"

    def fit(self, X, y=None):
        if not hasattr(self, "unknown_token") or self.unknown_token is None:
            self.unknown_token = "Unknown"
        return self

    @staticmethod
    def _norm(v):
        if v is None:
            return None
        if isinstance(v, float) and np.isnan(v):
            return None
        s = str(v).strip()
        if s == "" or s.lower() in {"nan", "none", "null"}:
            return None
        return s

    @staticmethod
    def _yn(v):
        if v is None:
            return None
        s = str(v).strip().lower()
        if s in {"yes", "y", "true", "1"}:
            return "Yes"
        if s in {"no", "n", "false", "0"}:
            return "No"
        return None

    def transform(self, X):
        X = X.copy()
        tok = getattr(self, "unknown_token", "Unknown") or "Unknown"

        yn_cols = [
            "psychiatric_disorder", "intellectual_disability",
            "cerebrovascular_disease", "head_trauma", "cns_infection",
            "substance_alcohol_abuse", "family_history",
        ]
        for c in yn_cols:
            if c in X.columns:
                X[c] = X[c].apply(self._yn).fillna(tok)

        cat_cols = ["sex", "seizure_type", "current_asm", "mri_lesion_type", "eeg_status_detail"]
        for c in cat_cols:
            if c in X.columns:
                X[c] = X[c].apply(self._norm).fillna(tok)

        return X


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Must match training engineered features.
    """

    comorb_cols = [
        "psychiatric_disorder",
        "intellectual_disability",
        "cerebrovascular_disease",
        "head_trauma",
        "cns_infection",
        "substance_alcohol_abuse",
        "family_history",
    ]

    @staticmethod
    def _yn_to_bin(v) -> int:
        if v is None:
            return 0
        s = str(v).strip().lower()
        return 1 if s in {"yes", "y", "true", "1"} else 0

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

        # numeric conversion
        for c in ["age", "age_of_onset", "pretreatment_seizure_count", "prior_asm_exposure_count"]:
            if c in X.columns:
                X[c] = pd.to_numeric(X[c], errors="coerce")

        # duration_since_onset
        if {"age", "age_of_onset"}.issubset(X.columns):
            X["duration_since_onset"] = (X["age"] - X["age_of_onset"]).clip(lower=0)

        # seizure burden log + high_seizure_burden (median threshold)
        if "pretreatment_seizure_count" in X.columns:
            sc = X["pretreatment_seizure_count"]
            X["seizure_burden_log"] = np.log1p(sc)
            X["high_seizure_burden"] = (sc > self.med_seizure_).astype(int)

        # seizure frequency risk
        if {"pretreatment_seizure_count", "age"}.issubset(X.columns):
            X["seizure_frequency_risk"] = X["pretreatment_seizure_count"] / (X["age"] + 1)

        # prior asm
        if "prior_asm_exposure_count" in X.columns:
            p = X["prior_asm_exposure_count"].fillna(0)
            X["high_prior_asm"] = (p > 1).astype(int)
            X["poly_asm_history"] = (p >= 3).astype(int)

        # age_of_onset_group (training bins)
        if "age_of_onset" in X.columns:
            o = X["age_of_onset"]
            X["age_of_onset_group"] = pd.cut(
                o,
                bins=[-0.01, 1, 18, 40, 65, 120],
                labels=["Infant", "Childhood", "Young_Adult", "Mid_Age", "Older"],
                include_lowest=True
            ).astype(str)

        # structural lesion flag
        if "mri_lesion_type" in X.columns:
            s = X["mri_lesion_type"].astype(str).str.strip().str.lower()
            s = s.replace({"nan": np.nan, "null": np.nan})
            X["structural_lesion_flag"] = (s.notna() & (s != "none") & (s != "normal") & (s != "unknown")).astype(int)

        # eeg epileptic flag
        if "eeg_status_detail" in X.columns:
            s = X["eeg_status_detail"].astype(str).str.strip().str.lower()
            s = s.replace({"nan": np.nan, "null": np.nan})
            X["eeg_epileptic_flag"] = (s.notna() & (s != "normal") & (s != "unknown")).astype(int)

        # comorbidity burden + *_bin columns + high_comorbidity
        existing = [c for c in self.comorb_cols if c in X.columns]
        if existing:
            for c in existing:
                X[c + "_bin"] = X[c].apply(self._yn_to_bin).astype(int)

            bin_cols = [c + "_bin" for c in existing]
            X["comorbidity_burden"] = X[bin_cols].sum(axis=1).astype(int)
            X["high_comorbidity"] = (X["comorbidity_burden"] > 0).astype(int)

        # clinical risk index
        risk_components = [c for c in [
            "high_seizure_burden",
            "structural_lesion_flag",
            "high_prior_asm",
            "high_comorbidity",
            "eeg_epileptic_flag",
        ] if c in X.columns]

        if risk_components:
            X["clinical_risk_index"] = X[risk_components].sum(axis=1).astype(int)

        return X


def _register_fake_module(module_name: str) -> types.ModuleType:
    if module_name not in sys.modules:
        sys.modules[module_name] = types.ModuleType(module_name)
    return sys.modules[module_name]


def register_pickle_shims() -> None:
    main_mod = _register_fake_module("main")
    setattr(main_mod, "ColumnNameCleaner", ColumnNameCleaner)
    setattr(main_mod, "CategoricalValueCleaner", CategoricalValueCleaner)
    setattr(main_mod, "FeatureEngineer", FeatureEngineer)

    main2 = _register_fake_module("__main__")
    setattr(main2, "ColumnNameCleaner", ColumnNameCleaner)
    setattr(main2, "CategoricalValueCleaner", CategoricalValueCleaner)
    setattr(main2, "FeatureEngineer", FeatureEngineer)

    try:
        from sklearn.calibration import CalibratedClassifierCV as _SkCalibratedClassifierCV
        fake_calib_mod = _register_fake_module("CalibratedClassifierCV")
        setattr(fake_calib_mod, "CalibratedClassifierCV", _SkCalibratedClassifierCV)
    except Exception:
        pass
