"""
FastAPI application factory and main app instance.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from judicaita import __version__
from judicaita.api.routes import analysis, documents, health
from judicaita.core.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.

    Handles startup and shutdown events for the FastAPI application.
    """
    # Startup
    settings = get_settings()
    app.state.settings = settings
    yield
    # Shutdown (cleanup if needed)


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        FastAPI: Configured application instance
    """
    # Create settings with debug=True if no API key, for development/testing
    import os

    if not os.environ.get("GOOGLE_API_KEY"):
        os.environ["DEBUG"] = "true"
        os.environ["GOOGLE_API_KEY"] = ""  # Empty but present

    settings = get_settings()

    app = FastAPI(
        title="JudicAIta API",
        description=(
            "Explainable Legal AI Assistant API. "
            "Provides document processing, reasoning trace generation, "
            "citation mapping, and plain-English summaries."
        ),
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(documents.router, prefix="/api/v1", tags=["Documents"])
    app.include_router(analysis.router, prefix="/api/v1", tags=["Analysis"])

    return app


# Create the default app instance
app = create_app()
