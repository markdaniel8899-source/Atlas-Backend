from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.services.blog.providers import (
    BlogProviderClient,
    BlogStage,
    ProviderChainError,
)
from app.services.blog.quotas import InMemoryQuotaLedger, Quota


class FakeHTTPError(RuntimeError):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [
            SimpleNamespace(message=SimpleNamespace(content=content))
        ]
        self.usage = SimpleNamespace(prompt_tokens=11, completion_tokens=7)


class FakeCompletions:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> FakeResponse:
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("Unexpected provider request.")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return FakeResponse(str(item))


class FakeClient:
    def __init__(self, responses: list[object]) -> None:
        self.completions = FakeCompletions(responses)
        self.chat = SimpleNamespace(completions=self.completions)


def make_settings(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "groq_blog_api_key": "groq-key",
        "groq_base_url": "https://groq.test/v1",
        "groq_blog_model": "openai/gpt-oss-120b",
        "gemini_api_key": "gemini-key",
        "gemini_base_url": "https://gemini.test/v1",
        "gemini_blog_model": "gemini-3.8-flash",
        "nvidia_api_key": "nvidia-key",
        "nvidia_base_url": "https://nvidia.test/v1",
        "nvidia_blog_model": "nvidia/nemotron-3-ultra-550b-a55b",
        "sambanova_api_key": "samba-key",
        "sambanova_base_url": "https://samba.test/v1",
        "sambanova_blog_model": "gpt-oss-120b",
        "openrouter_api_key": "openrouter-key",
        "openrouter_base_url": "https://openrouter.test/v1",
        "openrouter_blog_model": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "openrouter_auto_model": "openrouter/free",
        "request_timeout": 180.0,
        "blog_writer_timeout_seconds": 300.0,
        "site_url": "https://atlas.test",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


async def no_sleep(_: float) -> None:
    return None


def build_client(
    settings: SimpleNamespace,
    responses: dict[str, list[object]],
    *,
    ledger: InMemoryQuotaLedger | None = None,
) -> tuple[BlogProviderClient, dict[str, FakeClient]]:
    clients = {
        base_url: FakeClient(items) for base_url, items in responses.items()
    }

    def factory(
        api_key: str,
        base_url: str,
        timeout: float,
        headers: dict[str, str] | None,
    ) -> FakeClient:
        return clients[base_url]

    client = BlogProviderClient(
        settings,
        ledger=ledger,
        client_factory=factory,
        sleep=no_sleep,
    )
    return client, clients


class ProviderClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_writer_chain_order(self) -> None:
        settings = make_settings()
        client, _ = build_client(settings, {})
        targets = client.chain(BlogStage.WRITER)
        self.assertEqual(
            [(target.provider, target.model) for target in targets],
            [
                ("gemini", settings.gemini_blog_model),
                ("nvidia", settings.nvidia_blog_model),
                ("sambanova", settings.sambanova_blog_model),
                ("openrouter", settings.openrouter_blog_model),
                ("openrouter", settings.openrouter_auto_model),
            ],
        )

    async def test_falls_back_after_rate_limit(self) -> None:
        settings = make_settings()
        client, clients = build_client(
            settings,
            {
                settings.gemini_base_url: [
                    FakeHTTPError("rate limited", 429),
                    FakeHTTPError("rate limited", 429),
                ],
                settings.nvidia_base_url: ["hello"],
            },
        )
        result = await client.complete_text(
            BlogStage.WRITER, [{"role": "user", "content": "Write"}]
        )
        self.assertEqual(result.provider, "nvidia")
        self.assertEqual(result.text, "hello")
        self.assertEqual(
            [attempt.status for attempt in result.attempts],
            ["failed", "failed", "completed"],
        )
        self.assertEqual(len(clients[settings.gemini_base_url].completions.calls), 2)

    async def test_missing_key_is_skipped(self) -> None:
        settings = make_settings(gemini_api_key="")
        client, _ = build_client(
            settings, {settings.nvidia_base_url: ["fallback"]}
        )
        result = await client.complete_text(
            BlogStage.WRITER, [{"role": "user", "content": "Write"}]
        )
        self.assertEqual(result.provider, "nvidia")
        self.assertEqual(result.attempts[0].status, "skipped")
        self.assertIn("api key", result.attempts[0].error or "")

    async def test_non_retryable_error_moves_to_next_provider(self) -> None:
        settings = make_settings()
        client, clients = build_client(
            settings,
            {
                settings.gemini_base_url: [FakeHTTPError("bad request", 400)],
                settings.nvidia_base_url: ["next"],
            },
        )
        result = await client.complete_text(
            BlogStage.WRITER, [{"role": "user", "content": "Write"}]
        )
        self.assertEqual(result.provider, "nvidia")
        self.assertEqual(
            len(clients[settings.gemini_base_url].completions.calls), 1
        )

    async def test_invalid_json_falls_through_provider_chain(self) -> None:
        settings = make_settings(groq_blog_api_key="")

        def validate(data: dict[str, object]) -> object:
            return data["value"]

        client, _ = build_client(
            settings,
            {
                settings.gemini_base_url: ["not json", "still not json"],
                settings.sambanova_base_url: ['{"value": 7}'],
            },
        )
        result = await client.complete_json(
            BlogStage.TOPIC,
            [{"role": "user", "content": "Pick"}],
            validator=validate,
        )
        self.assertEqual(result.provider, "sambanova")
        self.assertEqual(result.data, 7)
        self.assertEqual(
            [attempt.status for attempt in result.attempts],
            ["skipped", "failed", "failed", "completed"],
        )

    async def test_json_mode_rejection_retries_without_response_format(
        self,
    ) -> None:
        settings = make_settings(groq_blog_api_key="")
        client, clients = build_client(
            settings,
            {
                settings.gemini_base_url: [
                    FakeHTTPError("response_format is not supported", 400),
                    '{"value": 9}',
                ]
            },
        )
        result = await client.complete_json(
            BlogStage.TOPIC,
            [{"role": "user", "content": "Pick"}],
            validator=lambda data: data["value"],
        )
        calls = clients[settings.gemini_base_url].completions.calls
        self.assertEqual(result.data, 9)
        self.assertIn("response_format", calls[0])
        self.assertNotIn("response_format", calls[1])

    async def test_quota_exhaustion_skips_provider(self) -> None:
        settings = make_settings()
        ledger = InMemoryQuotaLedger(
            {
                "gemini": Quota(daily_requests=0),
                "nvidia": Quota(daily_requests=10),
            }
        )
        client, _ = build_client(
            settings,
            {settings.nvidia_base_url: ["quota fallback"]},
            ledger=ledger,
        )
        result = await client.complete_text(
            BlogStage.WRITER, [{"role": "user", "content": "Write"}]
        )
        self.assertEqual(result.provider, "nvidia")
        self.assertEqual(result.attempts[0].status, "skipped")
        self.assertIn("quota", result.attempts[0].error or "")

    async def test_usage_is_recorded_once_per_request(self) -> None:
        settings = make_settings(groq_blog_api_key="")
        ledger = InMemoryQuotaLedger(
            {"gemini": Quota(daily_requests=10, daily_tokens=5000)}
        )
        client, _ = build_client(
            settings,
            {settings.gemini_base_url: ['{"value": 9}']},
            ledger=ledger,
        )
        await client.complete_json(
            BlogStage.TOPIC,
            [{"role": "user", "content": "Pick"}],
            validator=lambda data: data["value"],
        )
        usage = ledger.usage("gemini", settings.gemini_blog_model)
        self.assertEqual(usage.requests, 1)
        self.assertEqual(usage.tokens, 18)
        self.assertEqual(usage.errors, 0)

    async def test_failed_requests_are_recorded_once(self) -> None:
        settings = make_settings()
        ledger = InMemoryQuotaLedger(
            {"gemini": Quota(daily_requests=10)}
        )
        client, _ = build_client(
            settings,
            {
                settings.gemini_base_url: [FakeHTTPError("bad", 400)],
                settings.nvidia_base_url: [FakeHTTPError("bad", 400)],
                settings.sambanova_base_url: [FakeHTTPError("bad", 400)],
                settings.openrouter_base_url: [
                    FakeHTTPError("bad", 400),
                    FakeHTTPError("bad", 400),
                ],
            },
            ledger=ledger,
        )
        with self.assertRaises(ProviderChainError):
            await client.complete_text(
                BlogStage.WRITER, [{"role": "user", "content": "Write"}]
            )
        usage = ledger.usage("gemini", settings.gemini_blog_model)
        self.assertEqual(usage.requests, 1)
        self.assertEqual(usage.errors, 1)

    async def test_exhausted_chain_raises(self) -> None:
        settings = make_settings()
        responses = {
            settings.gemini_base_url: [FakeHTTPError("bad", 400)],
            settings.nvidia_base_url: [FakeHTTPError("bad", 400)],
            settings.sambanova_base_url: [FakeHTTPError("bad", 400)],
            settings.openrouter_base_url: [
                FakeHTTPError("bad", 400),
                FakeHTTPError("bad", 400),
            ],
        }
        client, _ = build_client(settings, responses)
        with self.assertRaises(ProviderChainError) as raised:
            await client.complete_text(
                BlogStage.WRITER, [{"role": "user", "content": "Write"}]
            )
        self.assertEqual(len(raised.exception.attempts), 5)


if __name__ == "__main__":
    unittest.main()
