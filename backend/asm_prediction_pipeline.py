from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
import warnings
import re

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score,
    brier_score_loss, confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

TARGET_COL = "seizure_freedom_12months"
RANDOM_STATE = 42

OUTCOME_DEFINITION = {
    "target_col": TARGET_COL,
    "clinical_standard": (
        "ILAE definition: sustained seizure freedom when patient is seizure-free for >1 year, "
        "OR has sporadic seizures separated by a period 3× the longest interval between seizures "
        "prior to treatment, whichever is longer."
    ),
    "note": (
        "If your dataset uses a simple '12 months since last seizure' rule without the "
        "3× interseizure-interval clause, then it is NOT ILAE-equivalent."
    ),
}

SUSPICIOUS_NAME_PATTERNS = [
    r"outcome", r"target", r"label", r"response",
    r"seizure[_\s]*free", r"seizure[_\s]*freedom",
    r"12[_\s]*month", r"12m", r"follow[_\s]*up", r"post", r"after",
    r"remission", r"success", r"failure"
]


def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(8):
        if (cur / "data").exists():
            return cur
        cur = cur.parent
    return start.resolve().parents[1]


HERE = Path(__file__).resolve()
REPO_ROOT = find_repo_root(HERE)

DEFAULT_CSV = REPO_ROOT / "data" / "ASM Prediction Dataset.csv"
DEFAULT_OUT_MODEL = REPO_ROOT / "models" / "asm_pipeline.joblib"
DEFAULT_REPORTS_DIR = REPO_ROOT / "models" / "reports"


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
    def fit(self, X, y=None):
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
    """
    Feature engineering + clinical risk index.
    """
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

    def fit(self, X, y=None):
        X = X.copy()

        # Median seizure count for "high_seizure_burden"
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

        # Derived features
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

        # Risk index
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


def _normalize_target(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    mapping = {"yes": 1, "y": 1, "true": 1, "1": 1, "no": 0, "n": 0, "false": 0, "0": 0}
    return s.map(mapping)


def suspicious_name_scan(columns: list[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        for pat in SUSPICIOUS_NAME_PATTERNS:
            if re.search(pat, col):
                rows.append((col, pat))
                break
    return pd.DataFrame(rows, columns=["column", "matched_pattern"])


def make_report_dir(base_dir: Path) -> Path:
    ts = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out = base_dir / ts
    out.mkdir(parents=True, exist_ok=True)
    return out


def make_preprocess():
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), make_column_selector(dtype_include=np.number)),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", ohe),
            ]), make_column_selector(dtype_include=["object"])),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def build_candidates():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=4000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=600, max_depth=14, min_samples_leaf=5,  # more stable probs
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=700, max_depth=5, learning_rate=0.04,
            subsample=0.85, colsample_bytree=0.85,
            eval_metric="logloss", random_state=RANDOM_STATE
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=700, learning_rate=0.04,
            num_leaves=31, subsample=0.85, colsample_bytree=0.85,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
    }


def build_model_pipelines(preprocess):
    pipes = {}
    for name, model in build_candidates().items():
        pipes[name] = Pipeline(steps=[
            ("valclean", CategoricalValueCleaner()),
            ("fe", FeatureEngineer(target_col=TARGET_COL)),
            ("preprocess", preprocess),
            ("model", model),
        ])
    return pipes


def stratified_cv_report(pipes: dict, X: pd.DataFrame, y: pd.Series, n_splits=5) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "roc_auc": "roc_auc",
        "avg_precision": "average_precision",
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "brier": "neg_brier_score",
    }

    rows = []
    for name, pipe in pipes.items():
        out = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
        row = {"Model": name}
        for k, vals in out.items():
            if not k.startswith("test_"):
                continue
            metric = k.replace("test_", "")
            arr = np.array(vals, dtype=float)
            if metric == "brier":
                arr = -arr
            row[f"{metric}_mean"] = float(arr.mean())
            row[f"{metric}_std"] = float(arr.std(ddof=1))
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["roc_auc_mean", "avg_precision_mean"], ascending=False)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    n = len(y_true)

    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def choose_threshold_from_oof_prob(oof_prob: np.ndarray, y_true: np.ndarray) -> float:
    thresholds = np.linspace(0.05, 0.95, 181)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        yhat = (oof_prob >= t).astype(int)
        f1v = f1_score(y_true, yhat, zero_division=0)
        if f1v > best_f1:
            best_f1, best_t = f1v, float(t)
    return best_t


def evaluate_on_dataset(model, X, y, threshold=0.5):
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, y_prob)),
        "avg_precision": float(average_precision_score(y, y_prob)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y, y_prob)),
        "ece": float(expected_calibration_error(y, y_prob, n_bins=10)),
    }, y_prob, y_pred


def save_curves(report_dir: Path, y_true: np.ndarray, y_prob: np.ndarray):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(report_dir / "roc_curve.png", dpi=180)
    plt.close()

    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(report_dir / "pr_curve.png", dpi=180)
    plt.close()

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    plt.figure(figsize=(7, 6))
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed fraction positive")
    plt.title("Calibration Plot")
    plt.tight_layout()
    plt.savefig(report_dir / "calibration_plot.png", dpi=180)
    plt.close()


def decision_curve_analysis(y_true: np.ndarray, y_prob: np.ndarray, report_dir: Path):
    thresholds = np.linspace(0.01, 0.99, 99)
    n = len(y_true)
    prevalence = float(np.mean(y_true))

    net_benefits = []
    treat_all = []
    treat_none = np.zeros_like(thresholds)

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        nb = (tp / n) - (fp / n) * (t / (1 - t))
        net_benefits.append(nb)

        nb_all = prevalence - (1 - prevalence) * (t / (1 - t))
        treat_all.append(nb_all)

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, net_benefits, label="Model")
    plt.plot(thresholds, treat_all, linestyle="--", label="Treat All")
    plt.plot(thresholds, treat_none, linestyle="--", label="Treat None")
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title("Decision Curve Analysis")
    plt.tight_layout()
    plt.savefig(report_dir / "decision_curve.png", dpi=180)
    plt.close()


def _training_ranges_and_categories(X_train: pd.DataFrame) -> dict:
    """
    Save basic training support info for reliability checks at inference time.
    """
    info = {"numeric_ranges": {}, "categorical_levels": {}}

    for col in X_train.columns:
        if pd.api.types.is_numeric_dtype(X_train[col]):
            v = pd.to_numeric(X_train[col], errors="coerce")
            info["numeric_ranges"][col] = {
                "min": float(np.nanmin(v)),
                "max": float(np.nanmax(v)),
                "p01": float(np.nanpercentile(v, 1)),
                "p99": float(np.nanpercentile(v, 99)),
            }
        else:
            # store cleaned levels
            s = X_train[col].astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True).str.replace(" ", "_", regex=False)
            s = s.replace({"nan": np.nan, "null": np.nan})
            levels = sorted(pd.Series(s.dropna().unique()).astype(str).tolist())
            info["categorical_levels"][col] = levels

    return info


def train_and_report(
    csv_path: Path,
    out_joblib_path: Path,
    reports_dir: Path,
    calibrate: bool = False,
    calibration_method: str = "sigmoid",  # SAFER default
):
    if not calibrate:
        raise RuntimeError(
            "Refusing to train without calibration. "
            "Run with --calibrate --calibration_method sigmoid (recommended)."
        )

    csv_path = Path(csv_path)
    out_joblib_path = Path(out_joblib_path)
    reports_dir = Path(reports_dir)

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    report_dir = make_report_dir(reports_dir)

    df_raw = pd.read_csv(csv_path)
    df = ColumnNameCleaner().fit_transform(df_raw).drop_duplicates().copy()

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column missing. Expected '{TARGET_COL}'.")

    y = _normalize_target(df[TARGET_COL])
    if y.isna().any():
        raise ValueError(
            f"Target produced {int(y.isna().sum())} NaNs after normalization. "
            f"Fix invalid target values in '{TARGET_COL}'."
        )
    y = y.astype(int)

    X = df.drop(columns=[TARGET_COL])

    suspicious_name_scan(list(X.columns)).to_csv(report_dir / "leakage_suspicious_names.csv", index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # For UI dropdown
    if "current_asm" in df.columns:
        tmp = CategoricalValueCleaner().transform(df[["current_asm"]])
        available_asms = sorted(tmp["current_asm"].dropna().astype(str).unique().tolist())
    else:
        available_asms = []

    preprocess = make_preprocess()
    pipes = build_model_pipelines(preprocess)

    cv_df = stratified_cv_report(pipes, X_train, y_train, n_splits=5)
    cv_df.to_csv(report_dir / "cv_metrics_train.csv", index=False)

    best_model_name = str(cv_df.iloc[0]["Model"])
    best_pipe = pipes[best_model_name]

    # ---- CALIBRATE FIRST (nested CV) ----
    # Use sigmoid by default to reduce overfitting extreme probabilities.
    calibrated = CalibratedClassifierCV(best_pipe, method=calibration_method, cv=5)

    # OOF probabilities of the CALIBRATED model (fixes your mismatch)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_prob = cross_val_predict(calibrated, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    best_threshold = choose_threshold_from_oof_prob(oof_prob, y_train.values)

    (report_dir / "best_threshold.txt").write_text(f"{best_threshold}\n", encoding="utf-8")

    # Fit final calibrated model on full training set
    calibrated.fit(X_train, y_train)
    final_label = f"{best_model_name}+calibrated({calibration_method})"

    test_metrics, y_prob_test, y_pred_test = evaluate_on_dataset(
        calibrated, X_test, y_test, threshold=best_threshold
    )

    save_curves(report_dir, y_test.values, y_prob_test)
    decision_curve_analysis(y_test.values, y_prob_test, report_dir)

    metrics_payload = {
        "model": final_label,
        "threshold": float(best_threshold),
        "outcome_definition": OUTCOME_DEFINITION,
        "test_metrics": test_metrics,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positive_rate_train": float(np.mean(y_train)),
        "positive_rate_test": float(np.mean(y_test)),
    }
    (report_dir / "test_metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    # Training-support metadata for reliability warnings
    support_info = _training_ranges_and_categories(X_train)

    artifact = {
        "model": calibrated,
        "model_name": final_label,
        "target_col": TARGET_COL,
        "threshold": float(best_threshold),
        "feature_columns_raw": X.columns.tolist(),
        "available_asms": available_asms,
        "test_metrics": test_metrics,
        "report_dir": str(report_dir),
        "outcome_definition": OUTCOME_DEFINITION,

        # guardrails (UPDATED)
        "guardrails": {
            # Global probability clip to avoid absurd outputs
            "global_prob_clip": {"min": 0.05, "max": 0.95},

            # Risk-aware caps based on clinical_risk_index (0-5)
            # If risk>=4 => cap 0.35, risk==3 => cap 0.55, else no cap.
            "risk_caps": [
                {"risk_min": 4, "cap": 0.35},
                {"risk_min": 3, "cap": 0.55},
            ],

            # Expanded "drug resistance" criteria (more realistic than >=100)
            "extreme_drug_resistance_cap": 0.30,
            "criteria": {
                "pretreatment_seizure_count_gte": 30,
                "prior_asm_exposure_count_gte": 3,
                "mri_lesion_type_not": "none",
                "eeg_status_detail_not": "normal",
            },
        },

        # reliability metadata for inference checks
        "training_support": support_info,
    }

    out_joblib_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out_joblib_path)

    print("\nBest model:", final_label)
    print("Threshold:", round(best_threshold, 3))
    print("Saved artifact:", out_joblib_path)
    print("Saved reports:", report_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=str(DEFAULT_CSV))
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT_MODEL))
    parser.add_argument("--reports", type=str, default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--calibration_method", type=str, default="sigmoid", choices=["sigmoid", "isotonic"])
    args = parser.parse_args()

    train_and_report(
        Path(args.csv),
        Path(args.out),
        Path(args.reports),
        calibrate=bool(args.calibrate),
        calibration_method=args.calibration_method,
    )
