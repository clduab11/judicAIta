"""
FastAPI dependencies for dependency injection.
"""

from typing import Annotated

from fastapi import Depends, Header, HTTPException

from judicaita.core.config import Settings, get_settings


def get_current_settings() -> Settings:
    """
    Get the current application settings.

    Returns:
        Settings: Application settings instance
    """
    return get_settings()


SettingsDep = Annotated[Settings, Depends(get_current_settings)]


async def get_api_key_from_header(
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> str | None:
    """
    Extract API key from request header.

    This is a placeholder for API key authentication.
    In production, this would validate the key against a database.

    Args:
        x_api_key: API key from X-API-Key header

    Returns:
        The API key if provided, None otherwise
    """
    return x_api_key


async def require_api_key(
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> str:
    """
    Require a valid API key for endpoint access.

    Raises HTTPException if no API key is provided.
    In production, this would also validate the key.

    Args:
        x_api_key: API key from X-API-Key header

    Returns:
        The validated API key

    Raises:
        HTTPException: If no API key is provided
    """
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required. Provide X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # TODO: Validate API key against database
    # For now, accept any non-empty key
    return x_api_key


APIKeyDep = Annotated[str, Depends(require_api_key)]
OptionalAPIKeyDep = Annotated[str | None, Depends(get_api_key_from_header)]


async def get_current_user(
    api_key: str = Depends(require_api_key),
) -> dict:
    """
    Get the current user based on API key.

    This is a placeholder that returns a mock user.
    In production, this would look up the user from the database.

    Args:
        api_key: Validated API key

    Returns:
        User information dictionary
    """
    # TODO: Look up user from database based on API key
    return {
        "user_id": "placeholder-user",
        "api_key": api_key,
        "is_active": True,
    }


CurrentUserDep = Annotated[dict, Depends(get_current_user)]
