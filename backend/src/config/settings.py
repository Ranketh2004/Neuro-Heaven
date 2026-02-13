"""Application settings and configuration."""
import os
from pathlib import Path
from datetime import timedelta
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # MongoDB
    MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    MONGO_DB_NAME: str = os.getenv("MONGO_DB_NAME", "neuroheaven")
    
    # JWT Configuration
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION: timedelta = timedelta(days=7)
    
    # API Configuration
    API_PREFIX: str = "/api/v1"
    
    class Config:
        # Load .env from parent directory (Neuro-Heaven root)
        env_file = str(Path(__file__).parent.parent.parent.parent / ".env")
        case_sensitive = True


settings = Settings()
