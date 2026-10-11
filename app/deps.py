"""Shared FastAPI dependencies for request authentication.

Every AI endpoint resolves the caller's Supabase session JWT here so that
anonymous traffic cannot burn model quota. Verification mirrors the pattern
used by the gamification router: the token is exchanged against Supabase
Auth with the service-role client, and any failure is a clean 401.
"""

from __future__ import annotations

import asyncio

from fastapi import Header, HTTPException
from supabase import Client, create_client

from app.config import get_settings


def _service_client() -> Client:
    settings = get_settings()
    if not settings.supabase_url or not settings.supabase_service_role_key:
        raise HTTPException(
            status_code=503,
            detail=(
                "Authentication is not configured: set SUPABASE_URL and "
                "SUPABASE_SERVICE_ROLE_KEY on the server."
            ),
        )
    return create_client(
        settings.supabase_url, settings.supabase_service_role_key
    )


def verify_bearer_token(authorization: str | None) -> str:
    """Resolve the caller's Supabase user id from a bearer JWT.

    Raises 401 when the header is missing/malformed or the token is
    invalid or expired.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token.")

    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Missing bearer token.")

    try:
        response = _service_client().auth.get_user(token)
    except Exception as exc:  # noqa: BLE001 - supabase-py raises various types
        raise HTTPException(
            status_code=401, detail="Invalid or expired session."
        ) from exc

    user = getattr(response, "user", None)
    if user is None or not getattr(user, "id", None):
        raise HTTPException(
            status_code=401, detail="Invalid or expired session."
        )
    return str(user.id)


async def get_current_user(
    authorization: str | None = Header(default=None),
) -> str:
    """Route dependency: require a valid Supabase session token.

    Returns the authenticated user's id. The Auth lookup runs in a worker
    thread so the sync supabase-py call never blocks the event loop.
    """
    return await asyncio.to_thread(verify_bearer_token, authorization)
