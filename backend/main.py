from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import logging

from src.routes.epi_diagnosis import epi_router

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
]

# Add CORS middleware to the FastAPI app
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(epi_router, prefix="/epilepsy_diagnosis", tags=["Epilepsy Diagnosis"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)