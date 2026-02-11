from __future__ import annotations

from typing import Optional
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase


def to_public_user(doc: dict) -> dict:
    return {
        "id": str(doc["_id"]),
        "full_name": doc["full_name"],
        "email": doc["email"],
    }


async def find_user_by_email(db: AsyncIOMotorDatabase, email: str) -> Optional[dict]:
    return await db["users"].find_one({"email": email})


async def find_user_by_id(db: AsyncIOMotorDatabase, user_id: str) -> Optional[dict]:
    try:
        oid = ObjectId(user_id)
    except Exception:
        return None
    return await db["users"].find_one({"_id": oid})


async def create_user(
    db: AsyncIOMotorDatabase,
    full_name: str,
    email: str,
    password_hash: str,
) -> dict:
    doc = {"full_name": full_name, "email": email, "password_hash": password_hash}
    res = await db["users"].insert_one(doc)
    created = await db["users"].find_one({"_id": res.inserted_id})
    if created is None:
        raise RuntimeError("User insert succeeded but document not found.")
    return created
