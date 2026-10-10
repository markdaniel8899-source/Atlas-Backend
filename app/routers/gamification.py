"""Gamification endpoints: achievement checks and avatar selection.

These wrap SECURITY DEFINER Postgres RPCs (migration 0011) with the
service-role client. The web app talks to Supabase directly and only uses
these routes as a server-side surface (automation, mobile clients, etc.);
both paths share the exact same database logic.

The social graph is friends only (friendships table + invite_friend RPC);
squad endpoints were removed with migration 0012.
"""

from __future__ import annotations

import re
from typing import Any

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field
from supabase import Client, create_client

from app.config import get_settings

router = APIRouter(prefix="/api", tags=["gamification"])

_DICEBEAR_PATH = re.compile(r"^/\d+\.x/[a-z-]+/(svg|png)$")


def _service_client() -> Client:
    settings = get_settings()
    if not settings.supabase_url or not settings.supabase_service_role_key:
        raise HTTPException(
            status_code=503,
            detail=(
                "Gamification service is not configured: set SUPABASE_URL "
                "and SUPABASE_SERVICE_ROLE_KEY on the server."
            ),
        )
    return create_client(
        settings.supabase_url, settings.supabase_service_role_key
    )


def _user_id_from(authorization: str | None) -> str:
    """Resolve the caller's Supabase user id from the bearer JWT."""
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
        raise HTTPException(status_code=401, detail="Invalid or expired session.")
    return str(user.id)


def _rpc(client: Client, function: str, **payload: Any) -> Any:
    result = client.rpc(function, payload).execute()
    return result.data


class EmptyRequest(BaseModel):
    """POST bodies that carry no data (kept for forward compatibility)."""


class AvatarSelectRequest(BaseModel):
    avatar_url: str | None = Field(default=None, max_length=500)


@router.post("/achievements/check")
async def check_achievements(
    _body: EmptyRequest | None = None,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """Run the achievement engine for the signed-in user.

    Called after quiz completion and on daily login; safe to call often -
    unlocks are idempotent (unique on user + achievement).
    """
    user_id = _user_id_from(authorization)
    data = _rpc(_service_client(), "check_user_achievements", p_user_id=user_id)
    payload = data if isinstance(data, dict) else {}
    return {
        "ok": bool(payload.get("ok", True)),
        "unlocked": payload.get("unlocked", []),
        "total": payload.get("total", 0),
    }


@router.post("/avatar/select")
async def select_avatar(
    body: AvatarSelectRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """Validate and store a DiceBear avatar URL on the caller's profile.

    Passing `avatar_url: null` clears the avatar (initials fallback).
    """
    user_id = _user_id_from(authorization)

    url = (body.avatar_url or "").strip()
    if url:
        if not _is_safe_avatar_url(url):
            raise HTTPException(
                status_code=422,
                detail="Only DiceBear avatar URLs are allowed.",
            )

    client = _service_client()
    client.table("profiles").update(
        {"avatar_url": url or None}
    ).eq("id", user_id).execute()

    return {"ok": True, "avatar_url": url or None}


def _is_safe_avatar_url(url: str) -> bool:
    from urllib.parse import urlparse

    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    return (
        parsed.scheme == "https"
        and parsed.hostname == "api.dicebear.com"
        and bool(_DICEBEAR_PATH.match(parsed.path))
    )
