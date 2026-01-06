from fastapi import APIRouter, UploadFile, File, HTTPException, Request
import os
import tempfile
import logging
from typing import Dict, Any


epi_router = APIRouter()
logger = logging.getLogger(__name__)

# =========================
# EXISTING FEATURE (KEEP)
# POST /epilepsy_diagnosis/predict
# =========================
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
            logger.info(f"Temporary file created at: {temp_filepath}")
            logger.info(f"File size: {len(content)} bytes")

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
async def predict_soz(
    request: Request,
    file: UploadFile = File(...)
) -> Dict[str, Any]:
    try:
        if not file.filename.lower().endswith(".edf"):
            raise HTTPException(status_code=400, detail="Only EDF files are supported for SOZ.")

        content = await file.read()

        # IMPORTANT: this service must be loaded in main.py startup:
        # app.state.soz_service = SOZInferenceService(...)
        if not hasattr(request.app.state, "soz_service"):
            raise HTTPException(
                status_code=500,
                detail="SOZ service not loaded. Check main.py startup load_models()."
            )

        soz_service = request.app.state.soz_service
        out = soz_service.predict_from_edf_bytes(
            edf_bytes=content,
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
        logger.error(f"SOZ prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"SOZ prediction error: {str(e)}")


@epi_router.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "ok"}
