# src/routes/auth.py
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo.errors import DuplicateKeyError, PyMongoError

from src.db.mongo import get_db
from src.db.user_repo import (
    find_user_by_email,
    find_user_by_id,
    create_user,
    to_public_user,
)
from src.schemas.auth import SignUpRequest, LoginRequest, TokenResponse, UserOut
from src.auth.security import hash_password, verify_password
from src.auth.jwt import create_access_token, decode_token

logger = logging.getLogger("neuroheaven")

router = APIRouter(prefix="/auth", tags=["Auth"])
bearer = HTTPBearer()


async def db_dep() -> AsyncIOMotorDatabase:
    """
    Dependency that returns a DB handle only if Mongo is reachable.
    This prevents random timeouts later and gives a consistent 503.
    """
    db = get_db()
    try:
        # Motor supports db.command("ping")
        await db.command("ping")
        return db
    except PyMongoError:
        logger.exception("MongoDB error (ping failed)")
        raise HTTPException(status_code=503, detail="Database unavailable")


@router.post("/signup", response_model=TokenResponse)
async def signup(payload: SignUpRequest, db: AsyncIOMotorDatabase = Depends(db_dep)):
    email = payload.email.lower().strip()

    try:
        existing = await find_user_by_email(db, email)
    except PyMongoError:
        logger.exception("MongoDB error during find_user_by_email")
        raise HTTPException(status_code=503, detail="Database unavailable")

    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    try:
        pw_hash = hash_password(payload.password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        created = await create_user(db, payload.full_name.strip(), email, pw_hash)
    except DuplicateKeyError:
        # Unique index violation (real duplicate)
        raise HTTPException(status_code=409, detail="Email already registered")
    except PyMongoError:
        # Real DB error (do NOT pretend it's a duplicate)
        logger.exception("MongoDB error during create_user")
        raise HTTPException(status_code=503, detail="Database unavailable")

    public = to_public_user(created)
    token = create_access_token({"sub": public["id"], "email": public["email"]})
    return {"access_token": token, "token_type": "bearer", "user": public}


@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest, db: AsyncIOMotorDatabase = Depends(db_dep)):
    email = payload.email.lower().strip()

    try:
        user = await find_user_by_email(db, email)
    except PyMongoError:
        logger.exception("MongoDB error during login lookup")
        raise HTTPException(status_code=503, detail="Database unavailable")

    if not user or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    public = to_public_user(user)
    token = create_access_token({"sub": public["id"], "email": public["email"]})
    return {"access_token": token, "token_type": "bearer", "user": public}


@router.get("/me", response_model=UserOut)
async def me(
    creds: HTTPAuthorizationCredentials = Depends(bearer),
    db: AsyncIOMotorDatabase = Depends(db_dep),
):
    payload = decode_token(creds.credentials)
    if not payload or "sub" not in payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    try:
        user = await find_user_by_id(db, payload["sub"])
    except PyMongoError:
        logger.exception("MongoDB error during find_user_by_id")
        raise HTTPException(status_code=503, detail="Database unavailable")

    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return to_public_user(user)
