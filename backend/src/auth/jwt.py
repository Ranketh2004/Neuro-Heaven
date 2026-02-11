from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError
from src.core.config import get_settings

def create_access_token(data: dict, expires_minutes: int | None = None) -> str:
    settings = get_settings()
    payload = data.copy()

    exp = datetime.now(timezone.utc) + timedelta(
        minutes=expires_minutes or settings.ACCESS_TOKEN_EXPIRE_MINUTES
    )
    payload["exp"] = exp

    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

def decode_token(token: str) -> dict | None:
    settings = get_settings()
    try:
        return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    except JWTError:
        return None
