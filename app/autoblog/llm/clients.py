"""LLM clients for AutoBlog v2 - Model role assignments."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

import requests
from requests.exceptions import HTTPError, Timeout
import google.generativeai as genai

from app.autoblog.config.settings import get_autoblog_settings


logger = logging.getLogger(__name__)


class BlogPipelineError(RuntimeError):
    """A pipeline stage failed in a way that aborts the run."""

    def __init__(self, message: str, *, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class JSONExtractResult:
    data: dict
    repaired: bool = False


def extract_json(text: str, prefer: str | tuple[str, ...] | None = None) -> JSONExtractResult:
    """Extract JSON from LLM output with basic repair."""
    text = text.strip()

    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        return JSONExtractResult(data=json.loads(text))
    except json.JSONDecodeError:
        pass

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


@dataclass(slots=True)
class ModelCallResult:
    text: str
    model: str
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: int


# ─── Groq (Research Model) ───────────────────────────────────────────────────


def _groq_research_chat(
    system: str,
    prompt: str,
    *,
    temperature: float,
    max_tokens: int,
    reasoning_effort: str = "low",
) -> ModelCallResult:
    """Sync Groq call for research tasks (queries, fact extraction, keywords, shortlist)."""
    settings = get_autoblog_settings()
    if not settings.groq_blog_api_key:
        raise BlogPipelineError("GROQ_BLOG_API_KEY is not configured.", status_code=503)

    start = time.perf_counter()
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
                "reasoning_effort": reasoning_effort,
            },
            timeout=120,
        )
        latency_ms = int((time.perf_counter() - start) * 1000)

        if response.status_code == 429 or response.status_code >= 500:
            if attempt == 0:
                logger.warning(f"[AutoBlog] Groq HTTP {response.status_code} - retrying once...")
                continue
            raise BlogPipelineError(
                f"Groq research call failed: HTTP {response.status_code}: {response.text[:200]}"
            )

        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]
        text = str(choice.get("message", {}).get("content") or "").strip()
        finish_reason = choice.get("finish_reason")
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")

        if text:
            logger.info(
                f"[AutoBlog] Groq research OK model={settings.groq_blog_model} "
                f"finish={finish_reason} prompt_tok={prompt_tokens} "
                f"completion_tok={completion_tokens} latency={latency_ms}ms"
            )
            return ModelCallResult(
                text=text,
                model=settings.groq_blog_model,
                finish_reason=finish_reason,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
            )
        logger.warning(f"[AutoBlog] Groq returned empty text (attempt {attempt + 1}/2) - retrying...")

    raise BlogPipelineError("Groq research returned empty text.")


async def groq_research(
    system: str,
    prompt: str,
    *,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    reasoning_effort: str = "low",
) -> ModelCallResult:
    """Async Groq research call with Nemotron fallback on failure."""
    try:
        return await asyncio.to_thread(
            _groq_research_chat,
            system,
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
    except (BlogPipelineError, HTTPError) as exc:
        logger.warning(
            f"Primary LLM failed with {type(exc).__name__}: {exc}. "
            "Triggering fallback..."
        )
        return await nemotron_chat(system, prompt, temperature=temperature, max_tokens=max_tokens)


# ─── NVIDIA NIM Nemotron (Editor-Planner + Writer) ───────────────────────────


def _nemotron_chat(
    system: str,
    prompt: str,
    *,
    temperature: float,
    max_tokens: int,
) -> ModelCallResult:
    """Sync NVIDIA NIM Nemotron call (planner + section writer)."""
    settings = get_autoblog_settings()
    if not settings.nvidia_api_key:
        raise BlogPipelineError("NVIDIA_API_KEY is not configured.", status_code=503)

    start = time.perf_counter()
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
            timeout=300,
        )
        latency_ms = int((time.perf_counter() - start) * 1000)

        if response.status_code == 429 or response.status_code >= 500:
            if attempt == 0:
                logger.warning(f"[AutoBlog] Nemotron HTTP {response.status_code} - retrying once...")
                continue
            raise BlogPipelineError(
                f"Nemotron call failed: HTTP {response.status_code}: {response.text[:160]}"
            )

        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]
        text = str(choice.get("message", {}).get("content") or "").strip()
        finish_reason = choice.get("finish_reason")
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")

        if text:
            logger.info(
                f"[AutoBlog] Nemotron OK model={settings.nvidia_model} "
                f"finish={finish_reason} prompt_tok={prompt_tokens} "
                f"completion_tok={completion_tokens} latency={latency_ms}ms"
            )
            return ModelCallResult(
                text=text,
                model=settings.nvidia_model,
                finish_reason=finish_reason,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
            )
        logger.warning(f"[AutoBlog] Nemotron returned empty text (attempt {attempt + 1}/2) - retrying...")

    raise BlogPipelineError("Nemotron returned empty text.")


async def nemotron_chat(
    system: str,
    prompt: str,
    *,
    temperature: float = 0.4,
    max_tokens: int = 4096,
) -> ModelCallResult:
    """Async Nemotron call with Gemini fallback on failure/timeout/short text."""
    try:
        result = await asyncio.to_thread(
            _nemotron_chat,
            system,
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # Check for too little text (heuristic: < 100 chars for writer calls)
        if len(result.text) < 100:
            raise BlogPipelineError("Nemotron returned too little text")
        return result
    except (BlogPipelineError, Timeout) as exc:
        if isinstance(exc, Timeout):
            logger.warning("Nemotron timed out, falling back to Gemini...")
        else:
            logger.warning(f"[AutoBlog] Nemotron failed, falling back to Gemini: {exc}")
        return await gemini_chat(system, prompt, temperature=temperature, max_tokens=max_tokens)


# ─── Gemini (Fallback + Meta/FAQ Final Pass) ─────────────────────────────────


def _gemini_model(system_instruction: str, model_name: str | None = None) -> genai.GenerativeModel:
    settings = get_autoblog_settings()
    if not settings.google_api_key:
        raise BlogPipelineError("GOOGLE_API_KEY is not configured.", status_code=503)
    genai.configure(api_key=settings.google_api_key)
    model = model_name or settings.gemini_model
    return genai.GenerativeModel(
        model_name=model,
        system_instruction=system_instruction,
    )


async def _generate_gemini(
    model: genai.GenerativeModel,
    prompt: str,
    *,
    temperature: float,
    max_tokens: int,
) -> ModelCallResult:
    start = time.perf_counter()
    try:
        response = await model.generate_content_async(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )
        latency_ms = int((time.perf_counter() - start) * 1000)
        text = (response.text or "").strip()
        finish_reason = getattr(response.candidates[0], "finish_reason", None) if response.candidates else None
        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", None) if usage else None
        completion_tokens = getattr(usage, "candidates_token_count", None) if usage else None

        if not text:
            raise BlogPipelineError("Gemini returned empty text.")

        logger.info(
            f"[AutoBlog] Gemini OK model={settings.gemini_model} "
            f"finish={finish_reason} prompt_tok={prompt_tokens} "
            f"completion_tok={completion_tokens} latency={latency_ms}ms"
        )
        return ModelCallResult(
            text=text,
            model=settings.gemini_model,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )
    except Exception as exc:
        raise BlogPipelineError(f"Gemini generation failed: {exc}") from exc


async def gemini_chat(
    system: str,
    prompt: str,
    *,
    temperature: float = 0.4,
    max_tokens: int = 4096,
) -> ModelCallResult:
    """Async Gemini call (final fallback). Uses gemini-3.8-flash as recommended by API."""
    model = _gemini_model(system, model_name="gemini-3.8-flash")
    return await _generate_gemini(model, prompt, temperature=temperature, max_tokens=max_tokens)


# ─── High-level Helpers ──────────────────────────────────────────────────────


async def groq_research_with_nemotron_fallback(
    system: str,
    prompt: str,
    *,
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> ModelCallResult:
    """Groq research -> Nemotron fallback (no Gemini for research)."""
    return await groq_research(system, prompt, temperature=temperature, max_tokens=max_tokens, reasoning_effort="low")


async def nemotron_with_gemini_fallback(
    system: str,
    prompt: str,
    *,
    temperature: float = 0.4,
    max_tokens: int = 4096,
) -> ModelCallResult:
    """Nemotron -> Gemini fallback."""
    return await nemotron_chat(system, prompt, temperature=temperature, max_tokens=max_tokens)


async def gemini_final_pass(
    system: str,
    prompt: str,
    *,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> ModelCallResult:
    """Gemini for meta/title/description/excerpt/FAQ final pass."""
    return await gemini_chat(system, prompt, temperature=temperature, max_tokens=max_tokens)