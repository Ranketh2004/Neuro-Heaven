from fastapi import APIRouter, UploadFile, File, HTTPException, Request
import io
import base64
import numpy as np
import nibabel as nib
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
import os
import tempfile
from typing import Dict, Any
import traceback


epi_router = APIRouter()
logger = logging.getLogger(__name__)

@epi_router.post("/predict")
async def predict_epilepsy(file: UploadFile = File(...)) -> Dict[str, Any]:
    temp_filepath = None

    try:
        if not file.filename.endswith('.edf'):
            raise HTTPException(status_code=400, detail="Only EDF files are supported.")
        
        logger.info(f"Processing file: {file.filename}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as temp_file:
            temp_filepath = temp_file.name
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            logger.info(f"Temporary file created at: {temp_filepath}")
            logger.info(f"File size: {len(content)} bytes")

            # try:
            #     logger.info("Initializing preprocessor...")
            #     preprocesser = EEGPreprocessor()
                
            #     logger.info("Starting preprocessing pipeline...")
            #     processed_data = preprocesser.run_pipeline(temp_filepath)
                
            #     logger.info(f"Preprocessing completed successfully!")
            #     logger.info(f"Processed data shape: {processed_data.shape}")
            #     print(f"Processed data shape: {processed_data.shape}")
                
            # except Exception as preprocess_error:
            #     logger.error(f"Preprocessing failed with error: {preprocess_error}")
            #     logger.error(f"Full traceback:\n{traceback.format_exc()}")
            #     raise HTTPException(
            #         status_code=422, 
            #         detail=f"Preprocessing error: {str(preprocess_error)}"
            #     )

        # Placeholder prediction results
        prediction = 1
        predictions = [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0]
        
        logger.info(f"Prediction completed for: {file.filename}")
        
        return {
            "prediction": prediction,
            "predictions": predictions
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing EDF file: {str(e)}")
    finally:
        if temp_filepath and os.path.exists(temp_filepath):
            try:
                os.unlink(temp_filepath)
                logger.info(f"Temporary file {temp_filepath} deleted.")
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_filepath}: {str(e)}")


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
