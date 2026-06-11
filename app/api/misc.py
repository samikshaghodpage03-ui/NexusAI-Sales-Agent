from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.db.session import get_db
from app.tools.catalog_tools import get_full_catalog
from app.models.schemas import CatalogResponse, HealthResponse
from app.config import get_settings

settings = get_settings()

catalog_router = APIRouter(tags=["catalog"])
health_router = APIRouter(tags=["health"])


@catalog_router.get("/catalog", response_model=CatalogResponse)
def get_catalog():
    """Returns the full product/pricing catalog."""
    return CatalogResponse(catalog=get_full_catalog())


@health_router.get("/health", response_model=HealthResponse)
def health_check(db: Session = Depends(get_db)):
    """Service health check — verifies DB connectivity."""
    try:
        db.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception as e:
        db_status = f"error: {str(e)}"

    return HealthResponse(
        status="ok" if db_status == "ok" else "degraded",
        version=settings.app_version,
        db=db_status,
    )