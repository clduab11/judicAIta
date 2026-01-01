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

    Note: This is a placeholder implementation. In production, this should
    validate the API key against the database. The current implementation
    accepts any non-empty key for development purposes only.

    Args:
        x_api_key: API key from X-API-Key header

    Returns:
        The validated API key

    Raises:
        HTTPException: If no API key is provided or invalid
    """
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required. Provide X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # TODO: Production implementation should:
    # 1. Hash the provided key
    # 2. Query the database for a matching key_hash
    # 3. Check if the key is active and not expired
    # 4. Update last_used_at timestamp
    # 5. Return the associated user info
    #
    # For development/demo purposes, accept any non-empty key.
    # This MUST be replaced before production deployment.
    import warnings

    warnings.warn(
        "API key validation is using placeholder implementation. "
        "Replace with database validation before production deployment.",
        UserWarning,
        stacklevel=2,
    )

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
