"""API routes initialization."""
from fastapi import APIRouter
from src.routes.epi_diagnosis import epi_router

__all__ = ["epi_router"]
