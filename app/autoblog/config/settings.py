"""AutoBlog v2 settings loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True, slots=True)
class AutoBlogSettings:
    """All AutoBlog configuration from environment."""

    # LLM providers
    nvidia_api_key: str
    nvidia_api_url: str
    nvidia_model: str

    google_api_key: str
    gemini_model: str

    groq_blog_api_key: str
    groq_base_url: str
    groq_blog_model: str

    # Research
    tavily_api_key: str

    # Images
    pixabay_api_key: str

    # Storage
    supabase_url: str
    supabase_service_role_key: str
    supabase_anon_key: str | None

    r2_endpoint_url: str | None
    r2_access_key_id: str | None
    r2_secret_access_key: str | None
    r2_bucket_name: str | None
    r2_public_url: str | None

    # Admin
    admin_secret: str

    # Behavior
    autoblog_draft: bool
    autoblog_max_posts_per_day: int
    autoblog_catchup_enabled: bool
    autoblog_schedule_hour: int
    autoblog_schedule_minute: int

    # Research params
    research_days: int
    research_max_results: int
    research_search_depth: str
    recent_posts_limit: int
    similarity_threshold: float
    topic_retry_limit: int

    # Writing params
    write_pause_seconds: int
    research_pause_seconds: int
    image_pause_seconds: int
    min_content_images: int
    max_content_images: int
    max_results: int

    # Publishing / ISR
    site_url: str = "https://atlas-frontend-pearl.vercel.app"
    vercel_revalidate_url: str = ""
    vercel_revalidate_secret: str = ""
    vercel_deploy_hook_url: str = ""

    @property
    def nemotron_api_key(self) -> str:
        """Preferred alias used by the dynamic pipeline."""
        return self.nvidia_api_key

    @property
    def gemini_api_key(self) -> str:
        """Preferred alias used by the dynamic pipeline."""
        return self.google_api_key

    @property
    def groq_research_api_key(self) -> str:
        """Groq key used for trend, keyword and table research calls."""
        return self.groq_blog_api_key

    @property
    def revalidation_ready(self) -> bool:
        """True when at least one Vercel revalidation hook is configured."""
        return bool(self.vercel_revalidate_url or self.vercel_deploy_hook_url)

    @property
    def blog_ready(self) -> bool:
        """True if all required keys for a full run are present."""
        return bool(
            self.groq_blog_api_key
            and self.supabase_url
            and self.supabase_service_role_key
        )

    @property
    def research_ready(self) -> bool:
        """True if research providers are configured."""
        return bool(
            self.tavily_api_key
            and (self.nvidia_api_key or self.groq_blog_api_key)
        )

    @property
    def images_ready(self) -> bool:
        """True if image providers are configured."""
        return bool(self.pixabay_api_key)

    @property
    def gemini_fallback_ready(self) -> bool:
        """True if Gemini fallback is configured."""
        return bool(self.google_api_key)


@lru_cache(maxsize=1)
def get_autoblog_settings() -> AutoBlogSettings:
    """Load and cache AutoBlog settings from environment."""
    return AutoBlogSettings(
        # NVIDIA NIM (NEMOTRON_* aliases take precedence)
        nvidia_api_key=(
            os.getenv("NEMOTRON_API_KEY", "").strip()
            or os.getenv("NVIDIA_API_KEY", "").strip()
        ),
        nvidia_api_url=os.getenv("NVIDIA_API_URL", "https://integrate.api.nvidia.com/v1").strip(),
        nvidia_model=(
            os.getenv("NEMOTRON_MODEL", "").strip()
            or os.getenv("NVIDIA_BLOG_MODEL", "nvidia/nemotron-3-ultra-550b-a55b").strip()
        ),

        # Google Gemini (GEMINI_* aliases take precedence)
        google_api_key=(
            os.getenv("GEMINI_API_KEY", "").strip()
            or os.getenv("GOOGLE_API_KEY", "").strip()
        ),
        gemini_model=(
            os.getenv("GEMINI_MODEL", "").strip()
            or os.getenv("BLOG_MODEL", "gemini-1.5-pro-latest").strip()
        ),

        # Groq (GROQ_API_KEY is accepted as a fallback for the dedicated blog key)
        groq_blog_api_key=(
            os.getenv("GROQ_BLOG_API_KEY", "").strip()
            or os.getenv("GROQ_API_KEY", "").strip()
        ),
        groq_base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").strip(),
        groq_blog_model=(
            os.getenv("GROQ_RESEARCH_MODEL", "").strip()
            or os.getenv("GROQ_BLOG_MODEL", "").strip()
            or "openai/gpt-oss-120b"
        ),

        # Tavily
        tavily_api_key=os.getenv("TAVILY_API_KEY", "").strip(),

        # Pixabay
        pixabay_api_key=os.getenv("PIXABAY_API_KEY", "").strip(),

        # Supabase
        supabase_url=os.getenv("SUPABASE_URL", "").strip(),
        supabase_service_role_key=os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip(),
        supabase_anon_key=os.getenv("SUPABASE_ANON_KEY", "").strip() or None,

        # R2
        r2_endpoint_url=os.getenv("R2_ENDPOINT_URL", "").strip() or None,
        r2_access_key_id=os.getenv("R2_ACCESS_KEY_ID", "").strip() or None,
        r2_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", "").strip() or None,
        r2_bucket_name=os.getenv("R2_BUCKET_NAME", "").strip() or None,
        r2_public_url=os.getenv("R2_PUBLIC_URL", "").strip() or None,

        # Admin
        admin_secret=os.getenv("ADMIN_SECRET", "").strip(),

        # Behavior flags
        autoblog_draft=os.getenv("AUTOBLOG_DRAFT", "1") == "1",
        autoblog_max_posts_per_day=int(os.getenv("AUTOBLOG_MAX_POSTS_PER_DAY", "1")),
        autoblog_catchup_enabled=os.getenv("AUTOBLOG_CATCHUP_ENABLED", "1") == "1",
        autoblog_schedule_hour=int(os.getenv("AUTOBLOG_SCHEDULE_HOUR", "9")),
        autoblog_schedule_minute=int(os.getenv("AUTOBLOG_SCHEDULE_MINUTE", "0")),

        # Research
        research_days=int(os.getenv("RESEARCH_DAYS", "3")),
        research_max_results=int(os.getenv("RESEARCH_MAX_RESULTS", "10")),
        research_search_depth=os.getenv("RESEARCH_SEARCH_DEPTH", "advanced"),
        recent_posts_limit=int(os.getenv("RECENT_POSTS_LIMIT", "10")),
        similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.70")),
        topic_retry_limit=int(os.getenv("TOPIC_RETRY_LIMIT", "2")),

        # Timing
        write_pause_seconds=int(os.getenv("WRITE_PAUSE_SECONDS", "3")),
        research_pause_seconds=int(os.getenv("RESEARCH_PAUSE_SECONDS", "2")),
        image_pause_seconds=int(os.getenv("IMAGE_PAUSE_SECONDS", "3")),
        min_content_images=int(os.getenv("MIN_CONTENT_IMAGES", "1")),
        max_content_images=int(os.getenv("MAX_CONTENT_IMAGES", "3")),
        max_results=int(os.getenv("MAX_RESULTS", "5")),

        # Publishing / ISR
        site_url=os.getenv("SITE_URL", "https://atlas-frontend-pearl.vercel.app").strip(),
        vercel_revalidate_url=os.getenv("VERCEL_REVALIDATE_URL", "").strip(),
        vercel_revalidate_secret=os.getenv("VERCEL_REVALIDATE_SECRET", "").strip(),
        vercel_deploy_hook_url=os.getenv("VERCEL_DEPLOY_HOOK_URL", "").strip(),
    )


def is_blog_configured() -> bool:
    """Quick check for health endpoint."""
    return get_autoblog_settings().blog_ready