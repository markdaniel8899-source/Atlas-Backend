from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any, Iterable, Literal, Sequence

from openai import AsyncOpenAI

from app.config import get_settings

Role = Literal["system", "user", "assistant"]
Message = dict[str, str]


GROQ = "groq"
GEMINI = "gemini"
ZEN = "zen"


def _resolve_credentials() -> dict[str, tuple[str, str]]:
    return {
        GROQ: (
            os.getenv("GROQ_API_KEY", "").strip(),
            os.getenv("GROQ_BASE_URL", "").strip() or "https://api.groq.com/openai/v1",
        ),
        GEMINI: (
            os.getenv("GEMINI_API_KEY", "").strip(),
            os.getenv("GEMINI_BASE_URL", "").strip()
            or "https://generativelanguage.googleapis.com/v1beta/openai",
        ),
        ZEN: (
            os.getenv("ZEN_API_KEY", "").strip(),
            os.getenv("ZEN_BASE_URL", "").strip() or "https://opencode.ai/zen/v1",
        ),
    }


CREDENTIALS = _resolve_credentials()
for _provider in (GROQ, GEMINI, ZEN):
    _key, _base = CREDENTIALS[_provider]
    print(
        f"Using API Key: {(_key[:5] if _key else '(not set)')}... "
        f"and Base URL: {_base} [{_provider}]"
    )


class Task(str, Enum):
    ROADMAP = "roadmap"
    QUIZ = "quiz"
    CODE = "code"
    CHAT = "chat"


# ROADMAP needs long JSON output -> Gemini free tier (250K TPM) handles it
# without the 8K TPM ceiling Groq hits. Everything else stays on Groq.
TASK_PROVIDER: dict[Task, str] = {
    Task.ROADMAP: GEMINI,
    Task.QUIZ: GROQ,
    Task.CODE: GROQ,
    Task.CHAT: GROQ,
}

MODEL_ROUTES: dict[Task, str] = {
    Task.ROADMAP: "gemini-3.5-flash",
    Task.QUIZ: "qwen/qwen3.8-27b",
    Task.CODE: "qwen/qwen3.8-27b",
    Task.CHAT: "openai/gpt-oss-20b",
}

FALLBACK_ROUTES: dict[Task, str] = {
    Task.ROADMAP: "openai/gpt-oss-120b",
    Task.QUIZ: "openai/gpt-oss-20b",
    Task.CODE: "openai/gpt-oss-20b",
    Task.CHAT: "openai/gpt-oss-120b",
}

TEMPERATURE: dict[Task, float] = {
    Task.ROADMAP: 0.5,
    Task.QUIZ: 0.6,
    Task.CODE: 0.2,
    Task.CHAT: 0.7,
}

MAX_TOKENS: dict[Task, int] = {
    Task.ROADMAP: 8192,
    Task.QUIZ: 4096,
    Task.CODE: 2048,
    Task.CHAT: 1200,
}


class AIServiceError(RuntimeError):
    def __init__(self, message: str, *, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class AIResult:
    text: str
    model: str
    task: Task


def pick_model(task: Task) -> str:
    return MODEL_ROUTES[task]


@lru_cache(maxsize=4)
def _client(provider: str) -> AsyncOpenAI:
    settings = get_settings()
    api_key, base_url = CREDENTIALS[provider]
    return AsyncOpenAI(
        api_key=api_key or "not-configured",
        base_url=base_url,
        timeout=settings.request_timeout,
        max_retries=max(settings.max_retries, 3),
    )


def _to_messages(messages: Sequence[dict[str, Any]] | Iterable[dict[str, Any]]) -> list[Message]:
    return [
        {"role": str(item["role"]), "content": str(item["content"])}
        for item in messages
    ]


async def _call(
    provider: str,
    model: str,
    messages: list[Message],
    *,
    temperature: float,
    max_tokens: int,
) -> str:
    api_key = CREDENTIALS[provider][0]
    if not api_key or api_key == "PASTE_YOUR_GEMINI_API_KEY_HERE":
        raise AIServiceError(
            f"{provider.upper()}_API_KEY is not configured on the server.",
            status_code=503,
        )

    extra: dict[str, Any] = {}
    if provider == GEMINI:
        # Keep thinking tokens small so the JSON fits in max_tokens.
        extra["reasoning_effort"] = "low"

    response = await _client(provider).chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.95,
        **extra,
    )
    choice = response.choices[0]
    content = (choice.message.content or "").strip()
    if not content:
        raise AIServiceError(f"{model} returned an empty response.")
    return content


async def complete(
    task: Task,
    messages: Sequence[dict[str, Any]] | Iterable[dict[str, Any]],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> AIResult:
    payload = _to_messages(messages)
    heat = TEMPERATURE[task] if temperature is None else temperature
    ceiling = MAX_TOKENS[task] if max_tokens is None else max_tokens

    candidates: list[tuple[str, str]] = [
        (TASK_PROVIDER[task], MODEL_ROUTES[task]),
        (GROQ, FALLBACK_ROUTES[task]),
    ]
    errors: list[str] = []
    for provider, model in dict.fromkeys(candidates):
        try:
            text = await _call(
                provider, model, payload, temperature=heat, max_tokens=ceiling
            )
            return AIResult(text=text, model=model, task=task)
        except AIServiceError as exc:
            errors.append(f"{provider}/{model}: {exc}")
        except Exception as exc:
            errors.append(f"{provider}/{model}: {type(exc).__name__}: {exc}")

    raise AIServiceError(f"AI request failed for task '{task.value}'. {'; '.join(errors)}")


def _close_constructs(fragment: str) -> str:
    """Close unterminated strings and unclosed `{`/`[` in a truncated fragment."""
    in_string = False
    escaped = False
    stack: list[str] = []
    for ch in fragment:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()

    out = fragment
    if in_string:
        if escaped:
            out = out[:-1]  # drop a dangling backslash
        out += '"'
    return out + "".join(reversed(stack))


def _repair_truncated_json(fragment: str) -> Any | None:
    """Salvage JSON that was cut off mid-stream (e.g. max_tokens truncation).

    Closes unterminated strings/brackets; if that still fails to parse, cuts
    back past the partially written tail member and retries.
    """
    current = fragment
    for _ in range(60):
        if not current.strip():
            return None
        current = current.rstrip()
        try:
            return json.loads(_close_constructs(current))
        except json.JSONDecodeError:
            pass
        cut = max(
            current.rfind(","),
            current.rfind(":"),
            current.rfind("{"),
            current.rfind("["),
        )
        if cut <= 0:
            return None
        current = current[:cut]
    return None


def extract_json(text: str, prefer: str | tuple[str, ...] | None = None) -> Any:
    response_text = text or ""
    print("RAW AI RESPONSE FOR PARSING:", response_text[:500])

    # 1) Strip markdown fences (```json ... ```) so only content remains.
    cleaned = re.sub(r"```[A-Za-z]*", "", response_text, flags=re.IGNORECASE).strip()
    decoder = json.JSONDecoder()
    parsed: Any = None

    # 2) First '{' ... last '}' cuts away any conversational text
    #    before/after the JSON block.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        candidate = cleaned[start : end + 1]
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            print("JSON PARSE FAILED:", exc)
            print("EXTRACTED BLOCK:", candidate[:500])

    # 2b) Truncated response (missing closing `}` / `]`): close the open
    #     constructs and drop the half-written tail member, then parse again.
    if parsed is None and start != -1:
        repaired = _repair_truncated_json(cleaned[start:])
        if repaired is not None:
            print("JSON REPAIRED FROM TRUNCATED RESPONSE")
            parsed = repaired
        else:
            print("JSON REPAIR FAILED. RAW RESPONSE:", response_text)

    # 3) Fallback: scan every '{' and take the first value that parses
    #    (preferring one that carries the required `prefer` key).
    if parsed is None or (prefer is not None and not _has_key(parsed, prefer)):
        first_ok = parsed
        for match in re.finditer(r"\{", cleaned):
            try:
                value, _ = decoder.raw_decode(cleaned[match.start() :])
            except json.JSONDecodeError:
                continue
            if first_ok is None:
                first_ok = value
            if prefer is None or _has_key(value, prefer):
                parsed = value
                break
        if parsed is None:
            parsed = first_ok

    if parsed is None:
        print("RAW AI RESPONSE (no JSON found):", response_text)
        raise AIServiceError("The model did not return parseable JSON.")

    # 4) Fallback validation for callers that require a specific structure.
    if prefer is not None and not _has_key(parsed, prefer):
        print("PARSED OBJECT (invalid structure):", json.dumps(parsed, ensure_ascii=False)[:500])
        raise AIServiceError(
            f"AI returned invalid structure. Raw output: {response_text[:100]}"
        )
    return parsed


def _has_key(value: Any, key: str | tuple[str, ...]) -> bool:
    """True when `value` (or a dict one level down) holds a non-empty list at `key`.

    `key` may be a tuple of alternatives; any one of them counts as a match.
    """
    wanted = (key,) if isinstance(key, str) else key
    if not isinstance(value, dict):
        return False

    def holds(mapping: dict[str, Any]) -> bool:
        return any(
            isinstance(mapping.get(name), list) and mapping.get(name)
            for name in wanted
        )

    if holds(value):
        return True
    return any(
        isinstance(nested, dict) and holds(nested) for nested in value.values()
    )


@dataclass(frozen=True, slots=True)
class AIJsonResult:
    data: Any
    model: str
    task: Task


async def complete_json(
    task: Task,
    messages: Sequence[dict[str, Any]] | Iterable[dict[str, Any]],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    prefer: str | tuple[str, ...] | None = None,
) -> AIJsonResult:
    result = await complete(task, messages, temperature=temperature, max_tokens=max_tokens)
    return AIJsonResult(data=extract_json(result.text, prefer=prefer), model=result.model, task=task)
