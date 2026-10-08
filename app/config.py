from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True, slots=True)
class Settings:
    groq_api_key: str
    groq_base_url: str
    allowed_origins: tuple[str, ...]
    request_timeout: float
    max_retries: int
    supabase_url: str
    supabase_anon_key: str

    @property
    def ai_ready(self) -> bool:
        return len(self.groq_api_key) > 0

    @property
    def supabase_ready(self) -> bool:
        return bool(self.supabase_url) and bool(self.supabase_anon_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    origins = tuple(
        origin.strip()
        for origin in os.getenv(
            "ALLOWED_ORIGINS", "http://localhost:5173"
        ).split(",")
        if origin.strip()
    )
    return Settings(
        groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
        groq_base_url=os.getenv(
            "GROQ_BASE_URL", "https://api.groq.com/openai/v1"
        ).strip(),
        allowed_origins=origins,
        request_timeout=float(os.getenv("AI_TIMEOUT_SECONDS", "180")),
        max_retries=int(os.getenv("AI_MAX_RETRIES", "1")),
        supabase_url=os.getenv("SUPABASE_URL", "").strip().rstrip("/"),
        supabase_anon_key=os.getenv("SUPABASE_ANON_KEY", "").strip(),
    )
