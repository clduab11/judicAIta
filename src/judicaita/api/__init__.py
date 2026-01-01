"""
JudicAIta API Package.

This package provides the FastAPI server implementation for JudicAIta,
exposing REST endpoints for document processing, analysis, and more.
"""

from judicaita.api.app import app, create_app

__all__ = ["app", "create_app"]
