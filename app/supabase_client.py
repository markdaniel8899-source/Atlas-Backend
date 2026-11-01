from __future__ import annotations

import httpx

from supabase import Client, ClientOptions, create_client

_http_client: httpx.Client | None = None

_HTTP_TIMEOUT = httpx.Timeout(180.0, connect=15.0)


def _get_http_client() -> httpx.Client:
    global _http_client
    if _http_client is None:
        _http_client = httpx.Client(
            http2=False,
            transport=httpx.HTTPTransport(http2=False, retries=0),
            timeout=_HTTP_TIMEOUT,
            follow_redirects=True,
        )
    return _http_client


def create_supabase_client(supabase_url: str, supabase_key: str) -> Client:
    return create_client(
        supabase_url,
        supabase_key,
        ClientOptions(httpx_client=_get_http_client()),
    )
