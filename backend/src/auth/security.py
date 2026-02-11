from passlib.context import CryptContext
import hashlib

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _prehash(password: str) -> str:
    """Deterministically reduce arbitrary-length passwords to a fixed-length
    representation using SHA-256 before passing to bcrypt. This avoids the
    72-byte input limit of bcrypt while keeping behaviour consistent for
    hashing and verification.
    """
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def hash_password(password: str) -> str:
    if not password or len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    pre = _prehash(password)
    return pwd_context.hash(pre)


def verify_password(password: str, password_hash: str) -> bool:
    pre = _prehash(password)
    return pwd_context.verify(pre, password_hash)
