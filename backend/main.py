from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import logging

from src.routes.epi_diagnosis import epi_router
from src.routes.auth.route import router as auth_router
from src.services.epi_diagnosis.soz_inference_service import SOZInferenceService
from src.services.epi_diagnosis.mri_inference_service import MRIFCDInferenceService
from src.config.database import MongoDatabase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI instance
app = FastAPI(
    title="NeuroHeaven EEG API",
    description="API for EEG, MRI, Clinical data processing and analysis",
    version="1.0.0"
)

# Define allowed origins for CORS
origins = [
    "http://localhost:3000",
    "http://localhost:8501",
]

# Add CORS middleware to the FastAPI app
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(
    epi_router,
    prefix="/epilepsy_diagnosis",
    tags=["Epilepsy Diagnosis"]
)
app.include_router(
    auth_router,
    prefix="/auth",
    tags=["Authentication"]
)

def load_soz_and_mri_services():
    """
    Expected folders:
      backend/src/models/soz/   -> SOZ artifacts
      backend/src/models/       -> MRI best.pt (or change path)
    """
   
    base_models_dir = Path(__file__).resolve().parent / "src" / "models"

    # -------- SOZ --------
    soz_models_dir = base_models_dir / "soz"
    logger.info(f"Loading SOZ service from: {soz_models_dir}")

    try:
        app.state.soz_service = SOZInferenceService(
            models_dir=soz_models_dir,   
            device="cpu"
        )
        logger.info("SOZ service loaded and ready.")
    except ModuleNotFoundError as e:
        logger.warning(f"SOZ optional dependency missing: {e}. SOZ endpoint will not work.")
    except Exception as e:
        logger.error(f"Failed to load SOZ service: {e}. SOZ endpoint will not work.")

    # -------- MRI --------
    try:
        mri_model_path = base_models_dir / "best.pt"  # change if your filename differs
        logger.info(f"Loading MRI FCD service from: {mri_model_path}")

        app.state.mri_service = MRIFCDInferenceService(
            model_pt_path=str(mri_model_path),
            img_size=192,
            base_ch=16,
        )
        logger.info("MRI FCD service loaded and ready.")
    except Exception as e:
        logger.error(f"Failed to load MRI service: {e}. MRI endpoint will not work.")


@app.on_event("startup")
def startup_event():
    """Initialize database connection on startup."""
    try:
        MongoDatabase.connect()
        logger.info("Application startup: MongoDB connection established")
    except Exception as e:
        logger.warning(f"MongoDB connection failed on startup: {e}")
        logger.info("App will continue, but auth endpoints will fail until DB connects")
    load_soz_and_mri_services()




@app.on_event("shutdown")
def shutdown_event():
    """Close database connection on shutdown."""
    MongoDatabase.disconnect()
    logger.info("Application shutdown: MongoDB connection closed")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)