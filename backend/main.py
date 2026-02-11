import logging
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------
# 1) Load .env BEFORE importing src.* modules
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neuroheaven")

# ---------------------------------------------------------
# 2) Now import FastAPI + your app modules
# ---------------------------------------------------------
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import get_settings
from src.routes.epi_diagnosis import epi_router
from src.routes.auth import router as auth_router

from src.services.epi_diagnosis.soz_inference_service import SOZInferenceService
from src.services.epi_diagnosis.mri_inference_service import MRIFCDInferenceService

# IMPORTANT: add ping_db in src.db.mongo
from src.db.mongo import ping_db, ensure_indexes, close_client


settings = get_settings()

app = FastAPI(
    title="NeuroHeaven EEG API",
    description="API for EEG, MRI, Clinical data processing and analysis",
    version="1.0.0",
)

# ---------------------------------------------------------
# 3) CORS from settings (not hardcoded)
# ---------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,  # <-- from config
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# 4) Routers
# ---------------------------------------------------------
app.include_router(auth_router)

app.include_router(
    epi_router,
    prefix="/epilepsy_diagnosis",
    tags=["Epilepsy Diagnosis"],
)

# ---------------------------------------------------------
# 5) Startup: Mongo ping -> indexes -> load models
# ---------------------------------------------------------
@app.on_event("startup")
async def startup():
    # ---- MongoDB: fail fast if URI/whitelist/DNS broken ----
    try:
        await ping_db()
        await ensure_indexes()
        logger.info("MongoDB connected + indexes ensured.")
    except Exception:
        logger.exception("MongoDB startup failed.")
        raise RuntimeError("MongoDB is not reachable. Fix Atlas/IP/URI before starting API.")


    # ---- Load models (keep your existing logic) ----
    models_dir = BASE_DIR / "src" / "models"
    logger.info(f"Loading services from: {models_dir}")

    try:
        app.state.soz_service = SOZInferenceService(models_dir=models_dir, device="cpu")
        logger.info("SOZ service loaded and ready.")
    except Exception:
        logger.exception("Failed to load SOZ service. Continuing without it.")
        app.state.soz_service = None

    try:
        mri_model_path = models_dir / "best.pt"
        logger.info(f"Loading MRI FCD service from: {mri_model_path}")
        app.state.mri_service = MRIFCDInferenceService(
            model_pt_path=str(mri_model_path),
            img_size=192,
            base_ch=16,
        )
        logger.info("MRI FCD service loaded and ready.")
    except Exception:
        logger.exception("Failed to load MRI service. Continuing without it.")
        app.state.mri_service = None


# ---------------------------------------------------------
# 6) Shutdown
# ---------------------------------------------------------
@app.on_event("shutdown")
def shutdown():
    close_client()
    logger.info("Mongo client closed.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
