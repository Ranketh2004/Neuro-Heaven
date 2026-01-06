# utils/asm_predictor.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, List

import numpy as np
import pandas as pd

# ✅ Correct import
from utils.model_loader import register_pickle_shims


def _patch_loaded_model(model: Any) -> None:
    if model is None:
        return

    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if step.__class__.__name__ == "CategoricalValueCleaner":
                if not hasattr(step, "unknown_token") or getattr(step, "unknown_token") is None:
                    setattr(step, "unknown_token", "Unknown")


def get_artifacts(model_path: str) -> Dict[str, Any]:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact file not found: {path.resolve()}")

    register_pickle_shims()

    obj = None
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

    model = obj.get("model")
    _patch_loaded_model(model)

    if obj.get("threshold") is None:
        obj["threshold"] = 0.50
    if not obj.get("model_name"):
        obj["model_name"] = obj.get("best_base_model") or type(model).__name__

    obj.setdefault("available_asms", [])
    obj.setdefault("feature_columns_raw", None)

    return obj


def _align_columns(X: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    if not cols:
        return X
    X = X.copy()
    for c in cols:
        if c not in X.columns:
            X[c] = np.nan
    return X[cols]


def _risk_index_from_inputs(x: Dict[str, Any]) -> int:
    score = 0

    ptsc = x.get("pretreatment_seizure_count")
    if ptsc is not None:
        if ptsc >= 20:
            score += 2
        elif ptsc >= 10:
            score += 1

    prior = x.get("prior_asm_exposure_count")
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


def _reliability_flags(x: Dict[str, Any]) -> List[str]:
    flags: List[str] = []

    age = x.get("age")
    onset = x.get("age_of_onset")
    ptsc = x.get("pretreatment_seizure_count")
    prior = x.get("prior_asm_exposure_count")

    if age is not None and (age < 0 or age > 120):
        flags.append("Age is outside plausible range (0–120).")

    if age is not None and onset is not None and onset > age:
        flags.append("Age of onset is greater than current age (inconsistent).")

    if ptsc is not None and ptsc > 50:
        flags.append("Pretreatment seizure count is unusually high (>50).")

    if prior is not None and prior > 50:
        flags.append("Prior ASM exposure count is unusually high (>50).")

    return flags


def predict(sample_patient: Dict[str, Any], artifacts: Dict[str, Any]) -> Tuple[int, float, int, List[str]]:
    model = artifacts.get("model")
    if model is None:
        raise ValueError("Artifacts missing 'model'.")

    threshold = float(artifacts.get("threshold", 0.5))

    # ✅ IMPORTANT: feed raw inputs; pipeline does the rest
    base = {k: (np.nan if v is None else v) for k, v in sample_patient.items()}
    X = pd.DataFrame([base])

    # ✅ Only align if feature_columns_raw exists AND is raw schema
    cols = artifacts.get("feature_columns_raw")
    if cols:
        X = _align_columns(X, cols)

    if not hasattr(model, "predict_proba"):
        raise TypeError("Loaded model does not support predict_proba().")

    prob = float(model.predict_proba(X)[0, 1])
    pred_label = int(prob >= threshold)

    risk_index = _risk_index_from_inputs(sample_patient)
    flags = _reliability_flags(sample_patient)

    return pred_label, prob, risk_index, flags
