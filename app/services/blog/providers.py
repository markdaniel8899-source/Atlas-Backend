from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Generic, Mapping, Sequence, TypeVar

from openai import AsyncOpenAI

from app.config import Settings, get_settings

from .quotas import InMemoryQuotaLedger, QuotaLedger

T = TypeVar("T")
Validator = Callable[[Any], T]
ClientFactory = Callable[[str, str, float, dict[str, str] | None], Any]


class BlogStage(str, Enum):
    TOPIC = "topic"
    KEYWORDS = "keywords"
    WRITER = "writer"
    VISUAL_PLACEMENT = "visual_placement"
    META = "meta"
    FAQ = "faq"


class ProviderChainError(RuntimeError):
    def __init__(
        self, stage: BlogStage, attempts: Sequence["ProviderAttempt"]
    ) -> None:
        self.stage = stage
        self.attempts = tuple(attempts)
        super().__init__(
            f"All providers failed for stage '{stage.value}' after "
            f"{len(self.attempts)} attempts."
        )


class InvalidProviderResponse(ValueError):
    pass


class ProviderEmptyResponse(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class StageSettings:
    temperature: float
    max_tokens: int


STAGE_SETTINGS: dict[BlogStage, StageSettings] = {
    BlogStage.TOPIC: StageSettings(0.5, 1_800),
    BlogStage.KEYWORDS: StageSettings(0.3, 1_600),
    BlogStage.WRITER: StageSettings(0.7, 8_000),
    BlogStage.VISUAL_PLACEMENT: StageSettings(0.35, 2_800),
    BlogStage.META: StageSettings(0.5, 700),
    BlogStage.FAQ: StageSettings(0.4, 2_000),
}


@dataclass(frozen=True, slots=True)
class ProviderTarget:
    provider: str
    model: str


@dataclass(frozen=True, slots=True)
class ProviderAttempt:
    provider: str
    model: str
    attempt: int
    status: str
    error: str | None = None
    duration_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True, slots=True)
class ProviderTextResult:
    text: str
    provider: str
    model: str
    stage: BlogStage
    attempts: tuple[ProviderAttempt, ...] = ()


@dataclass(frozen=True, slots=True)
class ProviderJsonResult(Generic[T]):
    data: T
    provider: str
    model: str
    stage: BlogStage
    attempts: tuple[ProviderAttempt, ...] = ()


@dataclass(frozen=True, slots=True)
class ProviderCredentials:
    api_key: str
    base_url: str


def _default_client_factory(
    api_key: str,
    base_url: str,
    timeout: float,
    default_headers: dict[str, str] | None,
) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=0,
        default_headers=default_headers,
    )


def provider_targets(
    settings: Settings | Any, stage: BlogStage
) -> list[ProviderTarget]:
    openrouter = [
        ProviderTarget("openrouter", settings.openrouter_blog_model),
        ProviderTarget("openrouter", settings.openrouter_auto_model),
    ]
    if stage is BlogStage.WRITER:
        targets = [
            ProviderTarget("gemini", settings.gemini_blog_model),
            ProviderTarget("nvidia", settings.nvidia_blog_model),
            ProviderTarget("sambanova", settings.sambanova_blog_model),
            *openrouter,
        ]
    elif stage in (BlogStage.TOPIC, BlogStage.KEYWORDS):
        targets = [
            ProviderTarget("groq", settings.groq_blog_model),
            ProviderTarget("gemini", settings.gemini_blog_model),
            ProviderTarget("sambanova", settings.sambanova_blog_model),
            *openrouter,
        ]
    elif stage is BlogStage.VISUAL_PLACEMENT:
        targets = [
            ProviderTarget("groq", settings.groq_blog_model),
            ProviderTarget("gemini", settings.gemini_blog_model),
            ProviderTarget("sambanova", settings.sambanova_blog_model),
            *openrouter,
        ]
    else:
        targets = [
            ProviderTarget("gemini", settings.gemini_blog_model),
            ProviderTarget("groq", settings.groq_blog_model),
            ProviderTarget("sambanova", settings.sambanova_blog_model),
            *openrouter,
        ]
    return list(dict.fromkeys(targets))


class BlogProviderClient:
    def __init__(
        self,
        settings: Settings | Any | None = None,
        *,
        ledger: QuotaLedger | None = None,
        client_factory: ClientFactory | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        max_attempts_per_provider: int = 2,
    ) -> None:
        self.settings = settings or get_settings()
        self.ledger = ledger or InMemoryQuotaLedger()
        self._client_factory = client_factory or _default_client_factory
        self._sleep = sleep or asyncio.sleep
        self._max_attempts = max(1, max_attempts_per_provider)
        self._clients: dict[str, Any] = {}
        self.last_provider: str | None = None
        self.last_model: str | None = None

    def chain(self, stage: BlogStage) -> list[ProviderTarget]:
        return provider_targets(self.settings, stage)

    def _credentials(self, provider: str) -> ProviderCredentials:
        if provider == "groq":
            return ProviderCredentials(
                self.settings.groq_blog_api_key, self.settings.groq_base_url
            )
        if provider == "gemini":
            return ProviderCredentials(
                self.settings.gemini_api_key, self.settings.gemini_base_url
            )
        if provider == "nvidia":
            return ProviderCredentials(
                self.settings.nvidia_api_key, self.settings.nvidia_base_url
            )
        if provider == "sambanova":
            return ProviderCredentials(
                self.settings.sambanova_api_key,
                self.settings.sambanova_base_url,
            )
        if provider == "openrouter":
            return ProviderCredentials(
                self.settings.openrouter_api_key,
                self.settings.openrouter_base_url,
            )
        raise ValueError(f"Unknown blog provider: {provider}")

    def _headers_for(self, provider: str) -> dict[str, str] | None:
        if provider != "openrouter":
            return None
        return {
            "HTTP-Referer": (
                getattr(self.settings, "site_url", "") or "https://atlas.app"
            ),
            "X-Title": "ATLAS Blog",
        }

    def _client_for(self, provider: str) -> Any:
        client = self._clients.get(provider)
        if client is not None:
            return client
        creds = self._credentials(provider)
        timeout = max(
            float(getattr(self.settings, "request_timeout", 180.0)),
            float(getattr(self.settings, "blog_writer_timeout_seconds", 300.0)),
        )
        client = self._client_factory(
            creds.api_key, creds.base_url, timeout, self._headers_for(provider)
        )
        self._clients[provider] = client
        return client

    async def complete_text(
        self,
        stage: BlogStage,
        messages: Sequence[Mapping[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ProviderTextResult:
        settings = STAGE_SETTINGS[stage]
        heat = settings.temperature if temperature is None else temperature
        ceiling = settings.max_tokens if max_tokens is None else max_tokens
        payload = self._messages(messages)
        attempts: list[ProviderAttempt] = []

        for target in self.chain(stage):
            success, target_attempts = await self._run_target(
                target,
                payload,
                stage=stage,
                use_json=False,
                validator=None,
                temperature=heat,
                max_tokens=ceiling,
            )
            attempts.extend(target_attempts)
            if success is not None:
                self.last_provider = target.provider
                self.last_model = target.model
                return ProviderTextResult(
                    text=success.text,
                    provider=target.provider,
                    model=target.model,
                    stage=stage,
                    attempts=tuple(attempts),
                )

        raise ProviderChainError(stage, attempts)

    async def complete_json(
        self,
        stage: BlogStage,
        messages: Sequence[Mapping[str, Any]],
        *,
        validator: Validator[T] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ProviderJsonResult[T]:
        settings = STAGE_SETTINGS[stage]
        heat = settings.temperature if temperature is None else temperature
        ceiling = settings.max_tokens if max_tokens is None else max_tokens
        payload = self._messages(messages)
        attempts: list[ProviderAttempt] = []

        for target in self.chain(stage):
            success, target_attempts = await self._run_target(
                target,
                payload,
                stage=stage,
                use_json=True,
                validator=validator,
                temperature=heat,
                max_tokens=ceiling,
            )
            attempts.extend(target_attempts)
            if success is not None:
                self.last_provider = target.provider
                self.last_model = target.model
                return ProviderJsonResult(
                    data=success.data,
                    provider=target.provider,
                    model=target.model,
                    stage=stage,
                    attempts=tuple(attempts),
                )

        raise ProviderChainError(stage, attempts)

    async def _run_target(
        self,
        target: ProviderTarget,
        messages: list[dict[str, str]],
        *,
        stage: BlogStage,
        use_json: bool,
        validator: Validator[Any] | None,
        temperature: float,
        max_tokens: int,
    ) -> tuple["_TargetSuccess | None", list[ProviderAttempt]]:
        attempts: list[ProviderAttempt] = []
        creds = self._credentials(target.provider)
        if not creds.api_key:
            attempts.append(
                ProviderAttempt(
                    provider=target.provider,
                    model=target.model,
                    attempt=1,
                    status="skipped",
                    error="api key not configured",
                )
            )
            return None, attempts

        json_mode = use_json
        estimated_tokens = self._estimate_tokens(messages) + max_tokens
        for attempt_no in range(1, self._max_attempts + 1):
            if not await self.ledger.allows(
                target.provider,
                target.model,
                estimated_tokens=estimated_tokens,
            ):
                attempts.append(
                    ProviderAttempt(
                        provider=target.provider,
                        model=target.model,
                        attempt=attempt_no,
                        status="skipped",
                        error="daily quota exhausted",
                    )
                )
                return None, attempts

            started = time.perf_counter()
            input_tokens = 0
            output_tokens = 0
            recorded = False
            try:
                response = await self._request(
                    target,
                    messages,
                    stage=stage,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                )
                text = self._response_text(response)
                input_tokens, output_tokens = self._usage(response)
                data = None
                if use_json:
                    data = self._parse_and_validate(text, validator)
                await self.ledger.record(
                    target.provider,
                    target.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
                recorded = True
                attempts.append(
                    ProviderAttempt(
                        provider=target.provider,
                        model=target.model,
                        attempt=attempt_no,
                        status="completed",
                        duration_ms=int(
                            (time.perf_counter() - started) * 1000
                        ),
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )
                )
                return _TargetSuccess(text=text, data=data), attempts
            except Exception as exc:
                if not recorded:
                    await self.ledger.record(
                        target.provider,
                        target.model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        error=True,
                    )
                attempts.append(
                    ProviderAttempt(
                        provider=target.provider,
                        model=target.model,
                        attempt=attempt_no,
                        status="failed",
                        error=f"{type(exc).__name__}: {exc}",
                        duration_ms=int(
                            (time.perf_counter() - started) * 1000
                        ),
                    )
                )
                if json_mode and self._json_mode_rejected(exc):
                    json_mode = False
                    if attempt_no < self._max_attempts:
                        continue
                    return None, attempts
                if (
                    self._retryable(exc)
                    and attempt_no < self._max_attempts
                ):
                    await self._sleep(0.5 * attempt_no)
                    continue
                return None, attempts

        return None, attempts

    async def _request(
        self,
        target: ProviderTarget,
        messages: list[dict[str, str]],
        *,
        stage: BlogStage,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> Any:
        client = self._client_for(target.provider)
        kwargs: dict[str, Any] = {
            "model": target.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.95,
            "timeout": (
                float(self.settings.blog_writer_timeout_seconds)
                if stage is BlogStage.WRITER
                else float(self.settings.request_timeout)
            ),
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if target.provider == "gemini":
            kwargs["reasoning_effort"] = "low"
        return await client.chat.completions.create(**kwargs)

    @staticmethod
    def _messages(
        messages: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, str]]:
        payload = [
            {
                "role": str(item["role"]),
                "content": str(item["content"]),
            }
            for item in messages
        ]
        if not payload:
            raise ValueError("Provider messages cannot be empty.")
        return payload

    @staticmethod
    def _estimate_tokens(messages: Sequence[Mapping[str, Any]]) -> int:
        return sum(len(str(item.get("content", ""))) for item in messages) // 4

    @staticmethod
    def _response_text(response: Any) -> str:
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise ProviderEmptyResponse("Provider returned no choices.")
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        text = content.strip() if isinstance(content, str) else ""
        if not text:
            raise ProviderEmptyResponse("Provider returned an empty response.")
        return text

    @staticmethod
    def _usage(response: Any) -> tuple[int, int]:
        usage = getattr(response, "usage", None)
        return (
            int(getattr(usage, "prompt_tokens", 0) or 0),
            int(getattr(usage, "completion_tokens", 0) or 0),
        )

    @staticmethod
    def _parse_and_validate(
        text: str, validator: Validator[Any] | None
    ) -> Any:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end <= start:
            raise InvalidProviderResponse("Response did not contain JSON.")
        try:
            data = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise InvalidProviderResponse(
                f"Invalid JSON: {exc.msg}"
            ) from exc
        if validator is None:
            return data
        try:
            return validator(data)
        except Exception as exc:
            raise InvalidProviderResponse(
                f"Schema validation failed: {exc}"
            ) from exc

    @staticmethod
    def _status_code(exc: Exception) -> int | None:
        value = getattr(exc, "status_code", None)
        if isinstance(value, int):
            return value
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
        return status if isinstance(status, int) else None

    @classmethod
    def _json_mode_rejected(cls, exc: Exception) -> bool:
        if cls._status_code(exc) != 400:
            return False
        text = str(exc).lower()
        return any(
            marker in text
            for marker in (
                "response_format",
                "json mode",
                "json_object",
                "not support",
            )
        )

    @classmethod
    def _retryable(cls, exc: Exception) -> bool:
        if isinstance(
            exc, (InvalidProviderResponse, ProviderEmptyResponse)
        ):
            return True
        status = cls._status_code(exc)
        if status is None:
            return True
        if status in (408, 409, 425, 429):
            return True
        if status >= 500:
            return True
        return False


@dataclass(frozen=True, slots=True)
class _TargetSuccess:
    text: str
    data: Any = None
