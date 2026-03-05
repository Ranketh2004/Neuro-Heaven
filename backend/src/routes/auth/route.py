"""Authentication routes for signup, login, and user profile."""
from fastapi import APIRouter, HTTPException, Depends, status, Header
from pymongo.database import Database
from typing import Optional
import logging

from src.config.database import get_database
from src.models.user import UserCreate, UserLogin, TokenResponse, UserResponse, GoogleLoginRequest
from src.services.auth_service import AuthService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Authentication"])


def get_auth_service(db: Database = Depends(get_database)) -> AuthService:
    """Dependency to get auth service."""
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection unavailable. Please check MongoDB connection."
        )
    return AuthService(db)


@router.post("/signup", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def signup(
    user_data: UserCreate,
    auth_service: AuthService = Depends(get_auth_service)
) -> dict:
    """
    Register a new user.
    
    - **full_name**: User's full name
    - **email**: User's email address (must be unique)
    - **password**: User's password (min 8 characters)
    """
    try:
        user = auth_service.signup(user_data)
        return user
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Signup error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during registration"
        )


@router.post("/login", response_model=TokenResponse)
def login(
    login_data: UserLogin,
    auth_service: AuthService = Depends(get_auth_service)
) -> dict:
    """
    Login user and receive JWT token.
    
    - **email**: User's email address
    - **password**: User's password
    """
    try:
        result = auth_service.login(login_data)
        return result
    except ValueError as e:
        logger.warning(f"Login validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Login error: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during login"
        )


@router.post("/google-login", response_model=TokenResponse)
def google_login(
    auth_data: GoogleLoginRequest,
    auth_service: AuthService = Depends(get_auth_service)
) -> dict:
    """
    Login or register user via Google OAuth.
    
    - **email**: User's email address
    - **full_name**: User's full name from Google profile
    - **picture**: User's profile picture URL (optional)
    - **google_id**: User's Google ID
    """
    try:
        result = auth_service.google_login(
            email=auth_data.email,
            full_name=auth_data.name,
            picture=auth_data.picture,
            google_id=auth_data.google_id
        )
        return result
    except Exception as e:
        logger.error(f"Google login error: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during Google login"
        )


@router.get("/me", response_model=UserResponse)
def get_current_user(
    authorization: Optional[str] = Header(None),
    auth_service: AuthService = Depends(get_auth_service)
) -> dict:
    """
    Get current logged-in user information.
    
    Requires Bearer token in Authorization header.
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No authentication token provided",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # Extract token from "Bearer <token>"
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token format",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    token = parts[1]
    
    # Verify token
    user_id = auth_service.verify_token(token)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # Get user from database
    user = auth_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Remove hashed password from response
    user.pop("hashed_password", None)
    
    return user
