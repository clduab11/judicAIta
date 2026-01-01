"""
Health check endpoints.
"""

from datetime import datetime
from typing import Any

from fastapi import APIRouter

from judicaita import __version__

router = APIRouter()


@router.get("/health")
async def health_check() -> dict[str, Any]:
    """
    Basic health check endpoint.

    Returns:
        Health status information
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": __version__,
    }


@router.get("/ready")
async def readiness_check() -> dict[str, Any]:
    """
    Readiness check endpoint.

    Verifies that the application is ready to handle requests.
    Checks dependencies and services.

    Returns:
        Readiness status with component checks
    """
    checks: dict[str, bool] = {}

    # Check if settings are available
    try:
        from judicaita.core.config import get_settings

        _ = get_settings()
        checks["settings"] = True
    except Exception:
        checks["settings"] = False

    # Check if document service is available
    try:
        from judicaita.document_input import DocumentInputService

        _ = DocumentInputService()
        checks["document_service"] = True
    except Exception:
        checks["document_service"] = False

    # Check if citation service is available
    try:
        from judicaita.citation_mapping import CitationMappingService

        _ = CitationMappingService()
        checks["citation_service"] = True
    except Exception:
        checks["citation_service"] = False

    all_ready = all(checks.values())

    return {
        "status": "ready" if all_ready else "not_ready",
        "timestamp": datetime.utcnow().isoformat(),
        "version": __version__,
        "checks": checks,
    }
