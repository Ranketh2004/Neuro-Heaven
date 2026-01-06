from __future__ import annotations

from pathlib import Path
import sys
import types
import joblib

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

TARGET_COL = "seizure_freedom_12months"


def _register_module_alias(module_name: str, FeatureEngineer, ColumnNameCleaner, CategoricalValueCleaner):
    mod = sys.modules.get(module_name)
    if mod is None:
        mod = types.ModuleType(module_name)
        sys.modules[module_name] = mod
    setattr(mod, "FeatureEngineer", FeatureEngineer)
    setattr(mod, "ColumnNameCleaner", ColumnNameCleaner)
    setattr(mod, "CategoricalValueCleaner", CategoricalValueCleaner)


def _ensure_pickle_compat():
    class ColumnNameCleaner(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):  # noqa
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
        def fit(self, X, y=None):  # noqa
            return self

        def transform(self, X):
            X = X.copy()
            obj_cols = X.select_dtypes(include=["object"]).columns
            for c in obj_cols:
                s = X[c].astype(str).str.strip().str.lower()
                s = s.replace({"nan": np.nan, "null": np.nan})
                s = s.str.replace(r"\s+", " ", regex=True).str.strip()
                s = s.str.replace(" ", "_", regex=False)
                X[c] = s
            return X

    class FeatureEngineer(BaseEstimator, TransformerMixin):
        def __init__(self, target_col: str = TARGET_COL):
            self.target_col = target_col
            self.comorb_cols = [
                "psychiatric_disorder", "intellectual_disability",
                "cerebrovascular_disease", "head_trauma",
                "cns_infection", "substance_alcohol_abuse", "family_history"
            ]

        @staticmethod
        def _norm_yes_no(v):
            v = str(v).strip().lower()
            return 1 if v in ["yes", "y", "true", "1"] else 0

        def fit(self, X, y=None):  # noqa
            X = X.copy()
            if "pretreatment_seizure_count" in X.columns:
                self.med_seizure_ = pd.to_numeric(X["pretreatment_seizure_count"], errors="coerce").median()
                if pd.isna(self.med_seizure_):
                    self.med_seizure_ = 0.0
            else:
                self.med_seizure_ = 0.0

            self.cat_modes_ = {}
            self.num_medians_ = {}
            for c in X.columns:
                if c == self.target_col:
                    continue
                if X[c].dtype == "O" or str(X[c].dtype).startswith("category"):
                    mode = X[c].mode(dropna=True)
                    self.cat_modes_[c] = mode.iloc[0] if len(mode) else "unknown"
                else:
                    self.num_medians_[c] = pd.to_numeric(X[c], errors="coerce").median()
            return self

        def transform(self, X):
            X = X.copy()

            if "mri_lesion_type" in X.columns:
                X["mri_lesion_type"] = X["mri_lesion_type"].fillna("none")

            for c, m in getattr(self, "cat_modes_", {}).items():
                if c in X.columns:
                    X[c] = X[c].fillna(m)

            for c, m in getattr(self, "num_medians_", {}).items():
                if c in X.columns:
                    X[c] = pd.to_numeric(X[c], errors="coerce").fillna(m)

            if {"age", "age_of_onset"}.issubset(X.columns):
                X["duration_since_onset"] = (X["age"] - X["age_of_onset"]).clip(lower=0)

            if "pretreatment_seizure_count" in X.columns:
                X["seizure_burden_log"] = np.log1p(X["pretreatment_seizure_count"])
                X["high_seizure_burden"] = (X["pretreatment_seizure_count"] > self.med_seizure_).astype(int)

            if {"pretreatment_seizure_count", "age"}.issubset(X.columns):
                X["seizure_frequency_risk"] = X["pretreatment_seizure_count"] / (X["age"] + 1)

            if "prior_asm_exposure_count" in X.columns:
                X["high_prior_asm"] = (X["prior_asm_exposure_count"] > 1).astype(int)
                X["poly_asm_history"] = (X["prior_asm_exposure_count"] >= 3).astype(int)

            if "age_of_onset" in X.columns:
                X["age_of_onset_group"] = pd.cut(
                    X["age_of_onset"],
                    bins=[-0.01, 1, 18, 40, 65, 120],
                    labels=["Infant", "Childhood", "Young_Adult", "Mid_Age", "Older"],
                    include_lowest=True
                ).astype(str)

            if "mri_lesion_type" in X.columns:
                X["structural_lesion_flag"] = (X["mri_lesion_type"] != "none").astype(int)

            if "eeg_status_detail" in X.columns:
                eeg_clean = X["eeg_status_detail"].astype(str).str.strip().str.lower()
                X["eeg_epileptic_flag"] = (eeg_clean != "normal").astype(int)

            existing = [c for c in self.comorb_cols if c in X.columns]
            if existing:
                for c in existing:
                    X[c + "_bin"] = X[c].apply(self._norm_yes_no)
                bin_cols = [c + "_bin" for c in existing]
                X["comorbidity_burden"] = X[bin_cols].sum(axis=1)
                X["high_comorbidity"] = (X["comorbidity_burden"] > 0).astype(int)

            risk_components = [
                c for c in
                ["high_seizure_burden", "structural_lesion_flag", "high_prior_asm", "high_comorbidity", "eeg_epileptic_flag"]
                if c in X.columns
            ]
            if risk_components:
                X["clinical_risk_index"] = X[risk_components].sum(axis=1)
                X["clinical_risk_index_weighted"] = X["clinical_risk_index"] * 2.5
                X.drop(columns=["clinical_risk_index"], inplace=True)

            return X

    # Register aliases for joblib compatibility
    FeatureEngineer.__module__ = "main"
    ColumnNameCleaner.__module__ = "main"
    CategoricalValueCleaner.__module__ = "main"
    _register_module_alias("main", FeatureEngineer, ColumnNameCleaner, CategoricalValueCleaner)

    FeatureEngineer.__module__ = "__main__"
    ColumnNameCleaner.__module__ = "__main__"
    CategoricalValueCleaner.__module__ = "__main__"
    _register_module_alias("__main__", FeatureEngineer, ColumnNameCleaner, CategoricalValueCleaner)


def load_artifacts(joblib_path: str) -> dict:
    path = Path(joblib_path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifacts not found: {joblib_path}")

    _ensure_pickle_compat()

    artifacts = joblib.load(path)

    required = ["model", "threshold", "feature_columns_raw", "target_col"]
    missing = [k for k in required if k not in artifacts]
    if missing:
        raise ValueError(f"Invalid artifacts file. Missing keys: {missing}")

    if not isinstance(artifacts["feature_columns_raw"], list) or len(artifacts["feature_columns_raw"]) == 0:
        raise ValueError("Invalid artifacts: feature_columns_raw must be a non-empty list")

    return artifacts
