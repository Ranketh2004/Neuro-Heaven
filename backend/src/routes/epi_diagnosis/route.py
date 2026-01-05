from fastapi import APIRouter, UploadFile, File, HTTPException
import os
from pathlib import Path
import tempfile
import logging
from typing import Dict, Any


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
            logger.info(f"Temporary file created at: {temp_filepath}")
            logger.info(f"File size: {len(content)} bytes")

        # Placeholder prediction results
        prediction = 1  # Overall prediction (0 or 1)
        predictions = [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0]  # Predictions for segments
        
        logger.info(f"Prediction completed for: {file.filename}")
        
        return {
            "prediction": prediction,
            "predictions": predictions
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {e}")
        raise HTTPException(status_code=500, 
                            detail=f"Error processing EDF file: {str(e)}"
                            )
    finally:
        if temp_filepath and os.path.exists(temp_filepath):
            try:
                os.unlink(temp_filepath)
                logger.info(f"Temporary file {temp_filepath} deleted.")
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_filepath}: {str(e)}")


@epi_router.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "ok"}
