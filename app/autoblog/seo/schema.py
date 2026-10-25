"""Schema generation for SEO (JSON-LD, Open Graph, Twitter Cards, Canonical)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from app.autoblog.config.settings import get_autoblog_settings
from app.autoblog.meta import MetaBundle
from app.autoblog.models import KeywordPlan
from app.autoblog.writer.dynamic import DynamicArticleDraft


def build_article_schema(
    article: DynamicArticleDraft,
    meta: MetaBundle,
    keywords: KeywordPlan,
    cover_image_url: str,
    site_url: str,
    author_name: str = "ATLAS Team",
    author_url: str = "",
    published_at: Optional[datetime] = None,
    updated_at: Optional[datetime] = None,
) -> dict:
    """Build complete JSON-LD schema for Article + FAQPage + BreadcrumbList."""

    pub_date = published_at or datetime.now(timezone.utc)
    mod_date = updated_at or datetime.now(timezone.utc)

    # Article schema
    article_schema = {
        "@context": "https://schema.org",
        "@type": "Article",
        "headline": meta.title,
        "description": meta.description,
        "image": cover_image_url,
        "datePublished": pub_date.isoformat(),
        "dateModified": mod_date.isoformat(),
        "author": {
            "@type": "Person",
            "name": author_name,
            "url": author_url or f"{site_url}/about",
        },
        "publisher": {
            "@type": "Organization",
            "name": "ATLAS",
            "logo": {
                "@type": "ImageObject",
                "url": f"{site_url}/logo.png",
            },
        },
        "mainEntityOfPage": {
            "@type": "WebPage",
            "@id": f"{site_url}/blog/{article.title.lower().replace(' ', '-')}",
        },
        "keywords": ", ".join(keywords.all_keywords()) if hasattr(keywords, 'all_keywords') else "",
    }

    # FAQ schema - extract from article if available
    faq_items = []

    faq_schema = {
        "@context": "https://schema.org",
        "@type": "FAQPage",
        "mainEntity": faq_items,
    }

    # Breadcrumb schema
    breadcrumb_schema = {
        "@context": "https://schema.org",
        "@type": "BreadcrumbList",
        "itemListElement": [
            {
                "@type": "ListItem",
                "position": 1,
                "name": "Home",
                "item": f"{site_url}/",
            },
            {
                "@type": "ListItem",
                "position": 2,
                "name": "Blog",
                "item": f"{site_url}/blog",
            },
            {
                "@type": "ListItem",
                "position": 3,
                "name": "Article",
                "item": f"{site_url}/blog/{{slug}}",
            },
        ],
    }

    # Combine all schemas
    return {
        "@context": "https://schema.org",
        "@graph": [
            article_schema,
            faq_schema,
            breadcrumb_schema,
        ],
    }


def build_open_graph_meta(
    article: DynamicArticleDraft,
    meta: MetaBundle,
    cover_image_url: str,
    site_url: str,
    slug: str,
) -> dict:
    """Build Open Graph meta tags."""
    return {
        "og:title": meta.title,
        "og:description": meta.description,
        "og:image": cover_image_url,
        "og:url": f"{site_url}/blog/{slug}",
        "og:type": "article",
        "og:site_name": "ATLAS",
        "og:locale": "en_US",
        "article:published_time": datetime.now(timezone.utc).isoformat(),
        "article:modified_time": datetime.now(timezone.utc).isoformat(),
        "article:author": "ATLAS Team",
        "article:section": "AI Learning",
        "article:tag": ", ".join(["AI", "learning", "students"]),
    }


def build_twitter_card_meta(
    article: DynamicArticleDraft,
    meta: MetaBundle,
    cover_image_url: str,
) -> dict:
    """Build Twitter Card meta tags."""
    return {
        "twitter:card": "summary_large_image",
        "twitter:title": meta.title,
        "twitter:description": meta.description,
        "twitter:image": cover_image_url,
        "twitter:creator": "@atlas_ai",
        "twitter:site": "@atlas_ai",
    }


def build_canonical_url(site_url: str, slug: str) -> str:
    """Build canonical URL."""
    return f"{site_url.rstrip('/')}/blog/{slug}"


def build_article_schema_simple(
    article: DynamicArticleDraft,
    meta: MetaBundle,
    keywords: Any,
    cover_image_url: str,
    site_url: str,
    slug: str,
    author_name: str = "ATLAS Team",
    author_url: str = "",
    published_at: Optional[Any] = None,
    updated_at: Optional[Any] = None,
) -> str:
    """Generate complete JSON-LD script tag content."""
    schema = build_article_schema(
        article, None, None, cover_image_url, site_url
    )
    return json.dumps(schema, ensure_ascii=False, separators=(',', ':'))


def build_all_meta(
    article: DynamicArticleDraft,
    meta: MetaBundle,
    keywords: Any,
    cover_image_url: str,
    site_url: str,
    slug: str,
    author_name: str = "ATLAS Team",
    author_url: str = "",
) -> dict:
    """Build all meta tags for the page."""
    settings = get_autoblog_settings()
    author_url = author_url or f"{site_url}/about"

    og = build_open_graph_meta(article, None, cover_image_url, site_url, "")
    twitter = build_twitter_card_meta(None, None, cover_image_url)
    canonical = f"{site_url.rstrip('/')}/blog/"
    schema_json = build_article_schema(None, None, None, cover_image_url, site_url)

    return {
        "title": meta.title,
        "description": meta.description,
        "canonical": canonical,
        "og": og,
        "twitter": twitter,
        "schema_json": schema_json,
        "meta_robots": "index, follow",
        "article_published_time": datetime.now(timezone.utc).isoformat(),
        "article_modified_time": datetime.now(timezone.utc).isoformat(),
        "article_author": "ATLAS Team",
        "article_section": "AI Learning",
        "article_tags": ", ".join(["AI", "learning", "students"]),
    }


def build_complete_schema(
    article: DynamicArticleDraft,
    meta: MetaBundle,
    keywords: KeywordPlan,
    cover_image_url: str,
    site_url: str,
    slug: str,
    author_name: str = "ATLAS Team",
    author_url: str = "",
    published_at: Optional[datetime] = None,
    updated_at: Optional[datetime] = None,
) -> dict:
    """Build complete schema.org JSON-LD."""
    return build_article_schema(
        article, meta, keywords, cover_image_url, site_url,
        author_name, author_url, published_at, updated_at
    )


__all__ = [
    "build_article_schema",
    "build_open_graph_meta",
    "build_twitter_card_meta",
    "build_canonical_url",
    "build_all_meta",
    "build_article_schema_simple",
    "build_complete_schema",
]