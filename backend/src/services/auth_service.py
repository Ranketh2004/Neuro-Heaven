"""Authentication service with JWT and password management."""
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import jwt
from jose.exceptions import JWTError, ExpiredSignatureError
import hashlib
from passlib.context import CryptContext
from pymongo.database import Database
from bson import ObjectId
import logging

from src.config.settings import settings
from src.models.user import UserCreate, UserLogin, UserResponse, UserInDB

logger = logging.getLogger(__name__)

# Password hashing context - use argon2 (more secure than bcrypt)
# Falls back to bcrypt if argon2 not available
try:
    pwd_context = CryptContext(schemes=["argon2", "bcrypt"], deprecated="auto")
except Exception:
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthService:
    """Authentication service for user management and JWT tokens."""
    
    def __init__(self, db: Database):
        """Initialize auth service with database connection."""
        self.db = db
        self.users_collection = db["users"]
        self._ensure_indexes()
    
    def _ensure_indexes(self) -> None:
        """Create database indexes for users collection."""
        try:
            self.users_collection.create_index("email", unique=True)
            logger.info("Database indexes created successfully")
        except Exception as e:
            logger.warning(f"Index creation warning: {e}")
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using argon2 or bcrypt."""
        try:
            return pwd_context.hash(password)
        except Exception:
            # If argon2 backend not available at runtime, fall back to bcrypt
            fallback = CryptContext(schemes=["bcrypt"], deprecated="auto")
            return fallback.hash(password)
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a plain password against its hashed version."""
        try:
            return pwd_context.verify(plain_password, hashed_password)
        except Exception:
            # Try bcrypt verification if argon2 backend is missing
            fallback = CryptContext(schemes=["bcrypt"], deprecated="auto")
            return fallback.verify(plain_password, hashed_password)
    
    def create_access_token(self, user_id: str) -> str:
        """Create JWT access token for user."""
        expires = datetime.utcnow() + settings.JWT_EXPIRATION
        payload = {
            "sub": str(user_id),
            "exp": expires,
            "iat": datetime.utcnow()
        }
        token = jwt.encode(
            payload,
            settings.JWT_SECRET_KEY,
            algorithm=settings.JWT_ALGORITHM
        )
        return token
    
    def verify_token(self, token: str) -> Optional[str]:
        """Verify JWT token and return user_id if valid."""
        try:
            payload = jwt.decode(
                token,
                settings.JWT_SECRET_KEY,
                algorithms=[settings.JWT_ALGORITHM]
            )
            user_id: str = payload.get("sub")
            if user_id is None:
                return None
            return user_id
        except ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except JWTError as e:
            logger.warning(f"Invalid token: {e}")
            return None
    
    def signup(self, user_data: UserCreate) -> Dict[str, Any]:
        """Register a new user."""
        # Check if user already exists
        existing_user = self.users_collection.find_one({"email": user_data.email})
        if existing_user:
            raise ValueError("User with this email already exists")
        
        # Create new user document
        user_doc = UserInDB(
            email=user_data.email,
            full_name=user_data.full_name,
            hashed_password=self.hash_password(user_data.password)
        )
        
        # Insert into database using dict with by_alias to get MongoDB format
        insert_dict = user_doc.model_dump(by_alias=True, exclude_unset=False)
        result = self.users_collection.insert_one(insert_dict)
        
        logger.info(f"User registered: {user_data.email}")
        
        # Format response with the inserted ID
        user_response = {
            "_id": str(result.inserted_id),
            "email": user_data.email,
            "full_name": user_data.full_name,
            "created_at": user_doc.created_at,
            "updated_at": user_doc.updated_at
        }
        return user_response
    
    def login(self, login_data: UserLogin) -> Dict[str, Any]:
        """Authenticate user and return token."""
        # Find user by email
        user_doc = self.users_collection.find_one({"email": login_data.email})
        if not user_doc:
            raise ValueError("Invalid email or password")
        
        # Verify password
        if not self.verify_password(login_data.password, user_doc["hashed_password"]):
            raise ValueError("Invalid email or password")
        
        # Generate token
        token = self.create_access_token(str(user_doc["_id"]))
        
        # Update last login
        self.users_collection.update_one(
            {"_id": user_doc["_id"]},
            {"$set": {"updated_at": datetime.utcnow()}}
        )
        
        logger.info(f"User logged in: {login_data.email}")
        
        # Format response
        user_response = {
            "_id": str(user_doc["_id"]),
            "email": user_doc["email"],
            "full_name": user_doc["full_name"],
            "created_at": user_doc["created_at"],
            "updated_at": user_doc["updated_at"]
        }
        
        return {
            "access_token": token,
            "token_type": "bearer",
            "user": user_response
        }
    
    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID."""
        try:
            user_doc = self.users_collection.find_one({"_id": ObjectId(user_id)})
            if not user_doc:
                return None
            
            # Remove sensitive data and return
            user_doc.pop("hashed_password", None)
            user_doc["_id"] = str(user_doc["_id"])
            return user_doc
        except Exception as e:
            logger.error(f"Error fetching user: {e}")
            return None
