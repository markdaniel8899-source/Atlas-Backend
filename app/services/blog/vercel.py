from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from app.config import Settings, get_settings


class DeployError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DeployResult:
    triggered: bool
    status_code: int | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "triggered": self.triggered,
            "status_code": self.status_code,
            "error": self.error,
        }


async def trigger_deploy(
    settings: Settings | Any | None = None,
    *,
    client: Any | None = None,
) -> DeployResult:
    current = settings or get_settings()
    url = (current.vercel_deploy_hook_url or "").strip()
    if not url:
        raise DeployError("VERCEL_DEPLOY_HOOK_URL is not configured.")

    payload = {"source": "atlas-blog-pipeline"}
    try:
        if client is not None:
            response = await client.post(url, json=payload)
        else:
            async with httpx.AsyncClient(timeout=15.0) as http:
                response = await http.post(url, json=payload)
    except Exception as exc:
        return DeployResult(
            triggered=False,
            error=f"{type(exc).__name__}: {exc}",
        )

    status_code = int(getattr(response, "status_code", 0) or 0)
    if 200 <= status_code < 300:
        return DeployResult(triggered=True, status_code=status_code)
    return DeployResult(
        triggered=False,
        status_code=status_code,
        error=f"Deploy hook returned HTTP {status_code}.",
    )
