"""LLM clients for AutoBlog v2 - NVIDIA NIM, Groq, Gemini."""

from __future__ import annotations

import asyncio
import requests
import google.generativeai as genai
from dataclasses import dataclass

from app.autoblog.config.settings import get_autoblog_settings


class BlogPipelineError(RuntimeError):
    """A pipeline stage failed in a way that aborts the run."""

    def __init__(self, message: str, *, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


class GroqTransientError(RuntimeError):
    """Groq rate limit (429) or server error (5xx) -> retry on Gemini."""


# ────────────────── NVIDIA NIM (Research: Topic + Keywords) ──────────────────


def _nim_chat(
    system: str,
    prompt: str,
    *,
    temperature: float,
    max_tokens: int,
) -> str:
    """Sync NVIDIA NIM chat call (OpenAI-compatible)."""
    settings = get_autoblog_settings()
    if not settings.nvidia_api_key:
        raise BlogPipelineError("NVIDIA_API_KEY is not configured.", status_code=503)

    for attempt in range(2):
        response = requests.post(
            f"{settings.nvidia_api_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.nvidia_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.nvidia_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=90,
        )
        if response.status_code == 429 or response.status_code >= 500:
            if attempt == 0:
                print(
                    f"[AutoBlog] NIM HTTP {response.status_code} - retrying once...",
                    flush=True,
                )
                continue
            raise BlogPipelineError(
                f"NVIDIA NIM call failed: HTTP {response.status_code}: {response.text[:160]}"
            )
        response.raise_for_status()
        data = response.json()
        text = str(data["choices"][0]["message"]["content"] or "").strip()
        if text:
            return text
        print("[AutoBlog] NIM returned empty text - retrying once...", flush=True)

    raise BlogPipelineError("NVIDIA NIM returned empty text.")


async def nim_chat(
    system: str,
    prompt: str,
    *,
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> str:
    """Async wrapper for NVIDIA NIM chat."""
    return await asyncio.to_thread(
        _nim_chat, system, prompt, temperature=temperature, max_tokens=max_tokens
    )


# ────────────────── Groq Blog (Meta + Article Writing) ────────────────────────


def _groq_blog_chat(
    system: str,
    prompt: str,
    *,
    temperature: float,
    max_tokens: int,
) -> str:
    """Sync Groq chat call on the DEDICATED blog key."""
    settings = get_autoblog_settings()
    if not settings.groq_blog_api_key:
        raise BlogPipelineError("GROQ_BLOG_API_KEY is not configured.", status_code=503)

    for attempt in range(2):
        response = requests.post(
            f"{settings.groq_base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.groq_blog_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.groq_blog_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=120,
        )
        if response.status_code == 429 or response.status_code >= 500:
            raise GroqTransientError(
                f"HTTP {response.status_code}: {response.text[:200]}"
            )
        response.raise_for_status()
        data = response.json()
        text = str(data["choices"][0]["message"]["content"] or "").strip()
        if text:
            return text
        print(
            f"[AutoBlog] Groq returned empty text (attempt {attempt + 1}/2) - retrying...",
            flush=True,
        )

    raise BlogPipelineError("Groq returned empty text.")


async def groq_with_gemini_fallback(
    system: str,
    prompt: str,
    *,
    temperature: float,
    max_tokens: int,
) -> str:
    """Groq first; on 429/5xx retry the SAME prompt via Gemini."""
    try:
        return await asyncio.to_thread(
            _groq_blog_chat,
            system,
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except GroqTransientError as exc:
        print(
            "[AutoBlog] Groq rate limit hit. Falling back to Gemini...",
            flush=True,
        )
        print(f"[AutoBlog] Groq error detail: {exc}", flush=True)
        settings = get_autoblog_settings()
        if not settings.google_api_key:
            raise BlogPipelineError(
                "Groq failed and GOOGLE_API_KEY (Gemini fallback) is not configured.",
                status_code=503,
            ) from exc
        model = _gemini_model(system)
        return await _generate_gemini(
            model,
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )


# ────────────────── Gemini (Fallback only) ────────────────────────────────────


def _gemini_model(system_instruction: str) -> genai.GenerativeModel:
    settings = get_autoblog_settings()
    if not settings.google_api_key:
        raise BlogPipelineError("GOOGLE_API_KEY is not configured.", status_code=503)
    genai.configure(api_key=settings.google_api_key)
    return genai.GenerativeModel(
        model_name=settings.gemini_model,
        system_instruction=system_instruction,
    )


async def _generate_gemini(
    model: genai.GenerativeModel,
    prompt: str,
    *,
    temperature: float,
    max_tokens: int,
) -> str:
    try:
        response = await model.generate_content_async(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )
        text = (response.text or "").strip()
    except Exception as exc:
        raise BlogPipelineError(f"Gemini generation failed: {exc}") from exc
    if not text:
        raise BlogPipelineError("Gemini returned empty text.")
    return text


# ────────────────── JSON Extraction Helper ──────────────────────────────────


@dataclass
class JSONExtractResult:
    data: dict
    repaired: bool = False


def extract_json(text: str, prefer: str | tuple[str, ...] | None = None) -> JSONExtractResult:
    """Extract JSON from LLM output with basic repair."""
    import json
    import re

    # Try to find JSON block
    text = text.strip()

    # Remove markdown fences
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Try direct parse
    try:
        return JSONExtractResult(data=json.loads(text))
    except json.JSONDecodeError:
        pass

    # Try to extract from {...} or [...]
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return JSONExtractResult(data=json.loads(brace_match.group()), repaired=True)
        except json.JSONDecodeError:
            pass

    bracket_match = re.search(r"\[.*\]", text, re.DOTALL)
    if bracket_match:
        try:
            return JSONExtractResult(data=json.loads(bracket_match.group()), repaired=True)
        except json.JSONDecodeError:
            pass

    raise BlogPipelineError(f"Could not parse JSON from LLM output: {text[:200]}")