from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import logging

from src.routes.epi_diagnosis import epi_router
from src.services.epi_diagnosis.soz_inference_service import SOZInferenceService

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

# ✅ Register routers
app.include_router(
    epi_router,
    prefix="/epilepsy_diagnosis",
    tags=["Epilepsy Diagnosis"]
)

# ✅ Load SOZ model + artifacts once at startup
@app.on_event("startup")
def load_soz_service():
    """
    Loads your GATv2 artifacts from:
      backend/src/models/
        - GATv2_best_state_dict.pt
        - META_graph_windows.csv
        - config.joblib
        - node_scaler.joblib
    """
    models_dir = Path(__file__).parent / "src" / "models"
    logger.info(f"Loading SOZ service from: {models_dir}")

    app.state.soz_service = SOZInferenceService(
        models_dir=models_dir,
        device="cpu"
    )

    logger.info("SOZ service loaded and ready.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
