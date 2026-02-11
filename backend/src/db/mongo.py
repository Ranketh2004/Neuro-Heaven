# src/db/mongo.py
from __future__ import annotations

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from src.core.config import get_settings

_client: AsyncIOMotorClient | None = None


def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        settings = get_settings()
        _client = AsyncIOMotorClient(
            settings.MONGO_URI,
            serverSelectionTimeoutMS=10000,
        )
    return _client


def get_db() -> AsyncIOMotorDatabase:
    settings = get_settings()
    return get_client()[settings.MONGO_DB]


async def ping_db() -> None:
    # Use DB-level ping (works fine for Motor)
    db = get_db()
    await db.command("ping")


async def ensure_indexes() -> None:
    db = get_db()
    await db["users"].create_index("email", unique=True, name="uniq_email")


def close_client() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None
