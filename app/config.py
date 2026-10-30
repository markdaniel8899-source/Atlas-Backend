from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


def _env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return default


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int, low: int, high: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(low, min(high, value))


def _float(name: str, default: float, low: float, high: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(low, min(high, value))


@dataclass(frozen=True, slots=True)
class Settings:
    groq_api_key: str
    groq_base_url: str
    groq_blog_api_key: str
    groq_blog_model: str
    allowed_origins: tuple[str, ...]
    request_timeout: float
    max_retries: int
    supabase_url: str
    supabase_anon_key: str
    supabase_service_role_key: str
    google_api_key: str
    blog_model: str
    tavily_api_key: str
    admin_secret: str

    gemini_api_key: str
    gemini_base_url: str
    gemini_blog_model: str
    nvidia_api_key: str
    nvidia_base_url: str
    nvidia_blog_model: str
    sambanova_api_key: str
    sambanova_base_url: str
    sambanova_blog_model: str
    openrouter_api_key: str
    openrouter_base_url: str
    openrouter_blog_model: str
    openrouter_auto_model: str
    site_url: str
    pixabay_api_key: str

    r2_endpoint_url: str
    r2_access_key_id: str
    r2_secret_access_key: str
    r2_bucket_name: str
    r2_public_url: str
    vercel_deploy_hook_url: str

    blog_enabled: bool
    blog_min_words: int
    blog_max_words: int
    blog_min_score: int
    blog_max_repairs: int
    blog_visual_word_threshold: int
    blog_writer_timeout_seconds: float
    blog_daily_article_limit: int
    blog_cron_enabled: bool
    blog_cron_hour_utc: int
    blog_cron_minute_utc: int

    @property
    def ai_ready(self) -> bool:
        return len(self.groq_api_key) > 0

    @property
    def supabase_ready(self) -> bool:
        return bool(self.supabase_url) and bool(self.supabase_anon_key)

    @property
    def r2_ready(self) -> bool:
        return all(
            (
                self.r2_endpoint_url,
                self.r2_access_key_id,
                self.r2_secret_access_key,
                self.r2_bucket_name,
                self.r2_public_url,
            )
        )

    @property
    def blog_writer_ready(self) -> bool:
        return any(
            (
                self.gemini_api_key,
                self.nvidia_api_key,
                self.sambanova_api_key,
                self.openrouter_api_key,
            )
        )

    @property
    def blog_ready(self) -> bool:
        """Legacy readiness flag kept for older health/API consumers."""
        return bool(self.groq_blog_api_key) and bool(
            self.supabase_service_role_key
        )

    @property
    def blog_pipeline_ready(self) -> bool:
        """Everything required to generate, store, and publish an article.

        Vercel is intentionally excluded: a deploy failure is non-fatal and
        the published post is still saved. `deploy_ready` surfaces that hook
        separately so rollout can report it without blocking drafts.
        """
        return bool(
            self.blog_enabled
            and self.supabase_url
            and self.supabase_service_role_key
            and self.tavily_api_key
            and self.pixabay_api_key
            and self.blog_writer_ready
            and self.r2_ready
        )

    @property
    def deploy_ready(self) -> bool:
        return bool(self.vercel_deploy_hook_url)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    origins = tuple(
        origin.strip()
        for origin in os.getenv(
            "ALLOWED_ORIGINS", "http://localhost:5173"
        ).split(",")
        if origin.strip()
    )
    gemini_api_key = _env("GEMINI_API_KEY", "GOOGLE_API_KEY")
    return Settings(
        groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
        groq_base_url=os.getenv(
            "GROQ_BASE_URL", "https://api.groq.com/openai/v1"
        ).strip(),
        # Dedicated blog writer key — deliberately NOT GROQ_API_KEY, which is
        # reserved for Chat/Quiz so blog traffic can never starve the app.
        groq_blog_api_key=_env("GROQ_BLOG_API_KEY", "GROQ_API_KEY"),
        groq_blog_model=os.getenv(
            "GROQ_BLOG_MODEL", "openai/gpt-oss-120b"
        ).strip()
        or "openai/gpt-oss-120b",
        allowed_origins=origins,
        request_timeout=_float("AI_TIMEOUT_SECONDS", 180.0, 10.0, 600.0),
        max_retries=_int("AI_MAX_RETRIES", 1, 0, 5),
        supabase_url=os.getenv("SUPABASE_URL", "").strip().rstrip("/"),
        supabase_anon_key=os.getenv("SUPABASE_ANON_KEY", "").strip(),
        supabase_service_role_key=os.getenv(
            "SUPABASE_SERVICE_ROLE_KEY", ""
        ).strip(),
        google_api_key=os.getenv("GOOGLE_API_KEY", "").strip(),
        blog_model=os.getenv("GOOGLE_BLOG_MODEL", "gemini-3.8-flash").strip()
        or "gemini-3.8-flash",
        tavily_api_key=os.getenv("TAVILY_API_KEY", "").strip(),
        admin_secret=os.getenv("ADMIN_SECRET", "").strip(),
        gemini_api_key=gemini_api_key,
        gemini_base_url=os.getenv(
            "GEMINI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        ).strip(),
        gemini_blog_model=os.getenv(
            "GEMINI_BLOG_MODEL", "gemini-3.8-flash"
        ).strip()
        or "gemini-3.8-flash",
        nvidia_api_key=_env("NVIDIA_API_KEY", "NEMOTRON_API_KEY"),
        nvidia_base_url=os.getenv(
            "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
        ).strip(),
        nvidia_blog_model=os.getenv(
            "NVIDIA_BLOG_MODEL", "nvidia/nemotron-3-ultra-550b-a55b"
        ).strip()
        or "nvidia/nemotron-3-ultra-550b-a55b",
        sambanova_api_key=_env("SAMBANOVA_API_KEY", "SAMBA_API_KEY"),
        sambanova_base_url=os.getenv(
            "SAMBANOVA_BASE_URL", "https://api.sambanova.ai/v1"
        ).strip(),
        sambanova_blog_model=os.getenv(
            "SAMBANOVA_BLOG_MODEL", "gpt-oss-120b"
        ).strip()
        or "gpt-oss-120b",
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", "").strip(),
        openrouter_base_url=os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        ).strip(),
        openrouter_blog_model=os.getenv(
            "OPENROUTER_BLOG_MODEL",
            "nvidia/nemotron-3-ultra-550b-a55b:free",
        ).strip()
        or "nvidia/nemotron-3-ultra-550b-a55b:free",
        openrouter_auto_model=os.getenv(
            "OPENROUTER_AUTO_MODEL", "openrouter/free"
        ).strip()
        or "openrouter/free",
        site_url=os.getenv(
            "SITE_URL", "https://atlas-frontend-pearl.vercel.app"
        ).strip().rstrip("/"),
        pixabay_api_key=os.getenv("PIXABAY_API_KEY", "").strip(),
        r2_endpoint_url=os.getenv("R2_ENDPOINT_URL", "").strip(),
        r2_access_key_id=os.getenv("R2_ACCESS_KEY_ID", "").strip(),
        r2_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", "").strip(),
        r2_bucket_name=os.getenv("R2_BUCKET_NAME", "").strip(),
        r2_public_url=os.getenv("R2_PUBLIC_URL", "").strip().rstrip("/"),
        vercel_deploy_hook_url=os.getenv(
            "VERCEL_DEPLOY_HOOK_URL", ""
        ).strip(),
        blog_enabled=_flag("BLOG_PIPELINE_ENABLED", True),
        blog_min_words=_int("BLOG_MIN_WORDS", 900, 300, 4000),
        blog_max_words=_int("BLOG_MAX_WORDS", 2200, 600, 8000),
        blog_min_score=_int("BLOG_MIN_SCORE", 90, 0, 100),
        blog_max_repairs=_int("BLOG_MAX_REPAIRS", 2, 0, 5),
        blog_visual_word_threshold=_int(
            "BLOG_VISUAL_WORD_THRESHOLD", 1200, 300, 10000
        ),
        blog_writer_timeout_seconds=_float(
            "BLOG_WRITER_TIMEOUT_SECONDS", 300.0, 30.0, 900.0
        ),
        blog_daily_article_limit=_int(
            "BLOG_DAILY_ARTICLE_LIMIT", 1, 1, 10
        ),
        blog_cron_enabled=_flag("BLOG_CRON_ENABLED", True),
        blog_cron_hour_utc=_int("BLOG_CRON_HOUR_UTC", 9, 0, 23),
        blog_cron_minute_utc=_int("BLOG_CRON_MINUTE_UTC", 0, 0, 59),
    )
