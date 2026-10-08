"""Fetch a course's own data (syllabus, roadmap progress, completed-topic
notes) from Supabase so a quiz can be planned from a course_id alone.

Every query runs through PostgREST with the caller's Supabase JWT, so RLS
guarantees the rows belong to that user. Notes are fetched only for the
course's COMPLETED topics - notes from other courses or topics can never
leak in.

Returns None when Supabase is not configured, the token is missing/expired,
the course does not exist, or nothing is visible.
"""

from __future__ import annotations

import html
import re

import httpx

from app.config import get_settings

REQUEST_TIMEOUT = 8.0

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def strip_html(value: str) -> str:
    text = html.unescape(_TAG_RE.sub(" ", value))
    return _WS_RE.sub(" ", text).strip()


def _clip(value: str, limit: int) -> str:
    value = value.strip()
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"


def _headers(user_token: str | None) -> dict[str, str]:
    settings = get_settings()
    headers = {
        "apikey": settings.supabase_anon_key,
        "Accept": "application/json",
    }
    if user_token:
        headers["Authorization"] = (
            user_token if user_token.lower().startswith("bearer ") else f"Bearer {user_token}"
        )
    return headers


async def _get(
    client: httpx.AsyncClient, table: str, params: dict[str, object]
) -> list[dict]:
    response = await client.get(f"/rest/v1/{table}", params=params)
    if response.status_code != 200:
        raise httpx.HTTPStatusError(
            f"{table} returned {response.status_code}",
            request=response.request,
            response=response,
        )
    data = response.json()
    return data if isinstance(data, list) else []


async def fetch_course_bundle(
    course_id: int, user_token: str | None
) -> dict | None:
    """Everything one course quiz plan needs, or None on any failure.

    Returns a dict with:
      course             - the courses row (id, title, description, domain,
                           status, progress_percentage; `domain` may be
                           absent before migration 0008 is applied)
      topics             - course_topics rows (id, title, status) in order
      roadmap_title      - title of the latest roadmap ("" when none)
      nodes              - its roadmap_nodes (title, status)
      completed_ids      - ids of completed course_topics (notes link here)
      completed_titles   - titles of every completed syllabus/roadmap topic
      notes              - notes whose topic_id is in completed_ids, newest
                           first (notes of completed topics only)
    """
    settings = get_settings()
    if not settings.supabase_ready or course_id <= 0:
        return None

    try:
        async with httpx.AsyncClient(
            base_url=settings.supabase_url,
            headers=_headers(user_token),
            timeout=REQUEST_TIMEOUT,
        ) as client:
            try:
                courses = await _get(
                    client,
                    "courses",
                    {
                        "id": f"eq.{course_id}",
                        "select": (
                            "id,title,description,domain,status,"
                            "progress_percentage"
                        ),
                    },
                )
            except Exception:  # noqa: BLE001 - courses.domain needs 0008
                courses = await _get(
                    client,
                    "courses",
                    {
                        "id": f"eq.{course_id}",
                        "select": "id,title,description,status,progress_percentage",
                    },
                )
            if not courses:
                return None
            course = courses[0]

            topics = await _get(
                client,
                "course_topics",
                {
                    "course_id": f"eq.{course_id}",
                    "select": "id,title,status",
                    "order": "position.asc",
                },
            )

            roadmaps = await _get(
                client,
                "roadmaps",
                {
                    "course_id": f"eq.{course_id}",
                    "select": "id,title",
                    "order": "created_at.desc",
                    "limit": "1",
                },
            )
            nodes: list[dict] = []
            roadmap_title = ""
            if roadmaps:
                roadmap_title = str(roadmaps[0].get("title") or "").strip()
                nodes = await _get(
                    client,
                    "roadmap_nodes",
                    {
                        "roadmap_id": f"eq.{roadmaps[0]['id']}",
                        "select": "title,status",
                        "order": "phase_number.asc,position.asc",
                    },
                )

            completed_ids = [
                t["id"]
                for t in topics
                if t.get("status") == "completed" and t.get("id") is not None
            ]

            completed_titles: list[str] = []
            seen: set[str] = set()
            for source in (topics, nodes):
                for row in source:
                    if row.get("status") != "completed":
                        continue
                    title = _clip(str(row.get("title") or ""), 100)
                    key = title.lower()
                    if not title or key in seen:
                        continue
                    seen.add(key)
                    completed_titles.append(title)

            notes: list[dict] = []
            if completed_ids:
                notes = await _get(
                    client,
                    "notes",
                    {
                        "select": "id,title,content,topic_id,course_id,updated_at",
                        "topic_id": f"in.({','.join(str(i) for i in completed_ids)})",
                        "order": "updated_at.desc",
                        "limit": "12",
                    },
                )
    except Exception:  # noqa: BLE001 - never break generation on fetch
        return None

    return {
        "course": course,
        "topics": topics,
        "roadmap_title": roadmap_title,
        "nodes": nodes,
        "completed_ids": completed_ids,
        "completed_titles": completed_titles,
        "notes": notes,
    }


async def save_course_domain(
    course_id: int, domain: str, user_token: str | None
) -> bool:
    """Persist the lazily classified `courses.domain` (migration 0008).

    Best effort: without the column (or update rights) the PATCH fails and
    the plan still carries the classified value for this request.
    """
    settings = get_settings()
    if not settings.supabase_ready or course_id <= 0 or not domain:
        return False
    try:
        async with httpx.AsyncClient(
            base_url=settings.supabase_url,
            headers=_headers(user_token),
            timeout=REQUEST_TIMEOUT,
        ) as client:
            response = await client.patch(
                "/rest/v1/courses",
                params={"id": f"eq.{course_id}"},
                json={"domain": domain},
            )
            return response.status_code in (200, 204)
    except Exception:  # noqa: BLE001 - lazy backfill must never break
        return False
