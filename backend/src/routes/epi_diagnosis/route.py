from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from pydantic import BaseModel
import logging
import os
import tempfile
import mne
from typing import Dict, Any, Optional, List

from src.services.epilepsy_pipeline_service import EpilepsyPipeline, CNN_MODEL_PATH, FEATURE_LAYER_NAME


epi_router = APIRouter()
logger = logging.getLogger(__name__)

@epi_router.post("/epilepsy/predict")
async def predict_epilepsy(file: UploadFile = File(...)) -> Dict[str, Any]:
    temp_filepath = None

    try:
        if not file.filename.endswith('.edf'):
            raise HTTPException(status_code=400, detail="Only EDF files are supported.")

        logger.info(f"Processing file: {file.filename}")

        # Write upload to a temp file so MNE can read it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
            temp_filepath = tmp.name
            content = await file.read()
            tmp.write(content)

        logger.info(f"Saved temp file ({len(content)} bytes) → {temp_filepath}")

        # Load EEG with MNE
        raw = mne.io.read_raw_edf(temp_filepath, preload=True, verbose=False)

        # Run full pipeline: preprocess → spectrograms → CNN features → classifier → diagnosis
        pipeline = EpilepsyPipeline()
        result = pipeline.diagnose(raw_obj=raw, layer_name=FEATURE_LAYER_NAME, file_name=file.filename)

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing EDF file: {str(e)}")
    finally:
        if temp_filepath and os.path.exists(temp_filepath):
            try:
                os.unlink(temp_filepath)
            except Exception as ex:
                logger.warning(f"Could not delete temp file {temp_filepath}: {ex}")


# =========================
# NEW FEATURE (ADD)
# POST /epilepsy_diagnosis/soz/predict
# =========================
@epi_router.post("/soz/predict")
async def predict_soz(request: Request, file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        if not file.filename.lower().endswith(".edf"):
            raise HTTPException(status_code=400, detail="Only EDF files are supported.")

        if not hasattr(request.app.state, "soz_service"):
            raise HTTPException(status_code=500, detail="SOZ service not loaded. Check app startup.")

        edf_bytes = await file.read()

        out = request.app.state.soz_service.predict_from_edf_bytes(
            edf_bytes=edf_bytes,
            filename=file.filename,
            tmin=0.0,
            window_sec=10.0,
        )

        if not out.get("ok", False):
            raise HTTPException(status_code=422, detail=out.get("error", "SOZ prediction failed."))

        return out

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("SOZ route failed")
        raise HTTPException(status_code=500, detail=str(e))


@epi_router.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "ok"}


# =========================
# ASM Response Prediction
# POST /epilepsy_diagnosis/asm/predict
# =========================
class ASMPredictionRequest(BaseModel):
    age: Optional[int] = None
    age_of_onset: Optional[int] = None
    pretreatment_seizure_count: Optional[int] = None
    prior_asm_exposure_count: Optional[int] = None

    sex: Optional[str] = None
    seizure_type: Optional[str] = None
    current_asm: Optional[str] = None
    mri_lesion_type: Optional[str] = None
    eeg_status_detail: Optional[str] = None

    psychiatric_disorder: Optional[str] = None
    intellectual_disability: Optional[str] = None
    cerebrovascular_disease: Optional[str] = None
    head_trauma: Optional[str] = None
    cns_infection: Optional[str] = None
    substance_alcohol_abuse: Optional[str] = None
    family_history: Optional[str] = None


class ASMPredictionResponse(BaseModel):
    pred_label: int
    result_text: str
    prob_final: float
    clinician_summary: str
    applicability_indicator: int
    shap: Dict[str, Any] = {}
    ml_details: Dict[str, Any] = {}


@epi_router.post("/asm/predict", response_model=ASMPredictionResponse)
async def predict_asm_response(request: Request, body: ASMPredictionRequest):
    """Predict ASM treatment response (seizure freedom at 12 months)."""
    try:
        if not hasattr(request.app.state, "asm_service"):
            raise HTTPException(
                status_code=500,
                detail="ASM prediction service not loaded. Check app startup logs.",
            )

        patient_dict = body.model_dump()

        # Strip placeholder values coming from the frontend
        for k, v in patient_dict.items():
            if isinstance(v, str) and v.strip().lower() in {"select", "select an option"}:
                patient_dict[k] = None

        result = request.app.state.asm_service.predict(patient_dict)

        return ASMPredictionResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ASM prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# -----------------------------
# MRI FCD prediction endpoint
# -----------------------------


@epi_router.post("/mri/predict")
async def predict_mri(request: Request, file: UploadFile = File(...)) -> Dict[str, Any]:
    """Accepts .nii / .nii.gz upload, runs preprocessing + Keras FCD model + Grad-CAM
    and returns a base64 PNG overlay plus prediction stats.
    """
    temp_flair = None
    try:
        fname = file.filename
        if not (fname.lower().endswith(".nii") or fname.lower().endswith(".nii.gz")):
            raise HTTPException(status_code=400, detail="Only .nii / .nii.gz files are supported for MRI prediction.")

        if not hasattr(request.app.state, "mri_service"):
            raise HTTPException(status_code=500, detail="MRI service not loaded. Check backend startup.")

        # write upload to temp file
        suffix = ".nii.gz" if fname.lower().endswith(".gz") else ".nii"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
            temp_flair = tf.name
            content = await file.read()
            tf.write(content)

        service = request.app.state.mri_service
        result = service.predict(flair_path=temp_flair, t1_path=None)

        return {
            "image_b64": result["image_b64"],
            "mri_b64": result["mri_b64"],
            "stats": {
                "fcd_probability": result["fcd_probability"],
                "prediction": result["prediction"],
                "best_slice_info": result["best_slice_info"],
                "num_patches": result["num_patches"],
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MRI prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_flair and os.path.exists(temp_flair):
            try:
                os.unlink(temp_flair)
            except Exception:
                pass
