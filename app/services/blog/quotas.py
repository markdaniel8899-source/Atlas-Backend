from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Callable, Protocol


@dataclass(frozen=True, slots=True)
class Quota:
    daily_requests: int
    daily_tokens: int | None = None
    per_model: bool = False


DEFAULT_QUOTAS: dict[str, Quota] = {
    "groq": Quota(daily_requests=1_000, daily_tokens=200_000),
    "gemini": Quota(daily_requests=500),
    "nvidia": Quota(daily_requests=200),
    "sambanova": Quota(
        daily_requests=20, daily_tokens=200_000, per_model=True
    ),
    "openrouter": Quota(daily_requests=50),
}


@dataclass(slots=True)
class UsageBucket:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    errors: int = 0

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class QuotaLedger(Protocol):
    async def allows(
        self,
        provider: str,
        model: str,
        *,
        estimated_tokens: int = 0,
    ) -> bool: ...

    async def record(
        self,
        provider: str,
        model: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        error: bool = False,
    ) -> None: ...


class InMemoryQuotaLedger:
    def __init__(
        self,
        quotas: dict[str, Quota] | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._quotas = dict(DEFAULT_QUOTAS if quotas is None else quotas)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._buckets: dict[tuple[str, str, date], UsageBucket] = {}

    def _today(self) -> date:
        return self._clock().astimezone(timezone.utc).date()

    def _key(self, provider: str, model: str) -> tuple[str, str, date]:
        quota = self._quotas.get(provider)
        model_key = model if quota and quota.per_model else "*"
        return (provider, model_key, self._today())

    async def allows(
        self,
        provider: str,
        model: str,
        *,
        estimated_tokens: int = 0,
    ) -> bool:
        quota = self._quotas.get(provider)
        if quota is None:
            return True
        bucket = self._buckets.get(self._key(provider, model), UsageBucket())
        if bucket.requests >= quota.daily_requests:
            return False
        if (
            quota.daily_tokens is not None
            and bucket.tokens + max(estimated_tokens, 0) > quota.daily_tokens
        ):
            return False
        return True

    async def record(
        self,
        provider: str,
        model: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        error: bool = False,
    ) -> None:
        if provider not in self._quotas:
            return
        key = self._key(provider, model)
        bucket = self._buckets.setdefault(key, UsageBucket())
        bucket.requests += 1
        bucket.input_tokens += max(input_tokens, 0)
        bucket.output_tokens += max(output_tokens, 0)
        if error:
            bucket.errors += 1

    def usage(self, provider: str, model: str) -> UsageBucket:
        bucket = self._buckets.get(self._key(provider, model))
        if bucket is None:
            return UsageBucket()
        return UsageBucket(
            requests=bucket.requests,
            input_tokens=bucket.input_tokens,
            output_tokens=bucket.output_tokens,
            errors=bucket.errors,
        )
