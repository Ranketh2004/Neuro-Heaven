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

#from services.epi_diagnosis.preprocessing_service import EEGPreprocessor


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


# -----------------------------
# MRI FCD prediction endpoint
# -----------------------------


def _make_overlay_png_b64(flair_path: str, mask_path: str) -> str:
    """Create a PNG overlay (flair grayscale + colored mask) and return base64 string.

    This builds an explicit RGB blend so the lesion area is a visible color
    (purple) instead of relying on matplotlib colormaps which sometimes
    produce washed-out results when composited.
    """
    img = nib.load(flair_path)
    flair = img.get_fdata(dtype=np.float32)

    mimg = nib.load(mask_path)
    mask = mimg.get_fdata(dtype=np.float32)

    # pick slice with maximal mask area
    per_slice = mask.sum(axis=(0, 1))
    if per_slice.max() > 0:
        zi = int(per_slice.argmax())
    else:
        zi = int(flair.shape[2] // 2)

    flair_slice = flair[:, :, zi]
    mask_slice = mask[:, :, zi]

    # normalized flair for display [0..255]
    fmin, fmax = np.percentile(flair_slice, (2, 98))
    disp = np.clip((flair_slice - fmin) / (fmax - fmin + 1e-9), 0, 1)
    base_img = (disp * 255).astype(np.uint8)

    # ensure mask is binary 0/1
    bin_mask = (mask_slice > 0.5).astype(np.uint8)

    # create RGB base and color mask (purple)
    h, w = base_img.shape
    rgb = np.stack([base_img, base_img, base_img], axis=2)
    color = np.array([181, 0, 202], dtype=np.uint8)  # purple-ish
    color_mask = np.zeros_like(rgb)
    color_mask[bin_mask == 1] = color

    alpha = 0.55
    comp = (rgb.astype(np.float32) * (1.0 - alpha) + color_mask.astype(np.float32) * alpha).astype(np.uint8)

    # Convert to PIL and save PNG
    pil = Image.fromarray(comp)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)

    img_b64 = base64.b64encode(buf.read()).decode("ascii")
    return img_b64


@epi_router.post("/mri/predict")
async def predict_mri(request: Request, file: UploadFile = File(...)) -> Dict[str, Any]:
    """Accepts .nii / .nii.gz upload, runs preprocessing + model (app.state.mri_service)
    and returns a base64 PNG overlay plus basic stats.
    """
    temp_flair = None
    temp_mask = None
    try:
        fname = file.filename
        if not (fname.lower().endswith(".nii") or fname.lower().endswith(".nii.gz")):
            raise HTTPException(status_code=400, detail="Only .nii / .nii.gz files are supported for MRI prediction.")

        if not hasattr(request.app.state, "mri_service"):
            raise HTTPException(status_code=500, detail="MRI service not loaded. Check backend startup.")

        # write flair to temp file
        suffix = ".nii.gz" if fname.lower().endswith(".gz") else ".nii"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
            temp_flair = tf.name
            content = await file.read()
            tf.write(content)

        # prepare temp path for mask
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tm:
            temp_mask = tm.name

        service = request.app.state.mri_service
        out = service.predict(flair_path=temp_flair, t1_path=None, out_mask_path=temp_mask)

        if out.get("mask_path") is None:
            raise HTTPException(status_code=500, detail="Model run failed to produce mask file.")

        # create PNG overlay base64
        img_b64 = _make_overlay_png_b64(temp_flair, out["mask_path"])

        return {"image_b64": img_b64, "stats": out}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MRI prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for p in (temp_flair, temp_mask):
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except Exception:
                    pass
