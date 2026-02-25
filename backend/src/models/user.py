"""User model and schemas for authentication."""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field, field_validator
from bson import ObjectId


class PyObjectId(ObjectId):
    """Pydantic ObjectId type."""
    
    @classmethod
    def _get_validators_(cls):
        yield cls.validate
    
    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError(f"Invalid ObjectId: {v}")
        return ObjectId(v)


class UserBase(BaseModel):
    """Base user schema."""
    email: EmailStr
    full_name: str


class UserCreate(UserBase):
    """User creation schema."""
    password: str = Field(..., min_length=8)
    
    @field_validator('password')
    def validate_password(cls, v):
        """Validate password is not too long for bcrypt."""
        if len(v.encode('utf-8')) > 512:
            raise ValueError('Password is too long (max 512 characters)')
        return v


class UserLogin(BaseModel):
    """User login schema."""
    email: EmailStr
    password: str


class GoogleLoginRequest(BaseModel):
    """Google OAuth login request schema."""
    email: EmailStr
    name: str = Field(..., alias="full_name")
    picture: Optional[str] = None
    google_id: str
    
    class Config:
        populate_by_name = True


class UserResponse(UserBase):
    """User response schema (no password)."""
    id: Optional[str] = Field(alias="_id")
    created_at: datetime
    updated_at: datetime
    
    class Config:
        populate_by_name = True


class TokenResponse(BaseModel):
    """Token response schema."""
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class UserInDB(BaseModel):
    """User document structure in MongoDB."""
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    email: str
    full_name: str
    hashed_password: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        populate_by_name = True