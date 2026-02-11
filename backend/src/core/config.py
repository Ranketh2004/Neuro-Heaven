from functools import lru_cache
import os
from dotenv import load_dotenv

# Load .env from current working directory (backend/)
load_dotenv()

class Settings:
    MONGO_URI: str = os.getenv("MONGO_URI", "").strip()
    MONGO_DB: str = os.getenv("MONGO_DB", "neuroheaven").strip()

    SECRET_KEY: str = os.getenv("NH_SECRET_KEY", "CHANGE_ME").strip()
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("NH_TOKEN_EXPIRE_MIN", "60"))

    # Comma-separated origins
    CORS_ORIGINS: list[str] = [
        o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:8501,http://localhost:3000").split(",")
        if o.strip()
    ]

@lru_cache
def get_settings() -> Settings:
    s = Settings()
    if not s.MONGO_URI:
        raise RuntimeError("MONGO_URI is missing. Check your .env file.")
    return s
