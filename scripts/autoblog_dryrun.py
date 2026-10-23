#!/usr/bin/env python3
"""AutoBlog dry run script for testing article generation without publishing.

Usage:
    python scripts/autoblog_dryrun.py [--topic TOPIC] [--count N] [--output FILE]

Runs the pipeline in dry-run mode: generates article(s) but does not save to Supabase.
Outputs detailed diagnostics including article type, data availability, word count,
visuals kept/dropped with reasons, photo count, and gate pass/fail.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.services.auto_blog import (
    run_staggered_blog_pipeline,
    _research_and_select_topic,
    _fetch_recent_posts,
    _analyze_data_availability,
    _build_article_profile,
    _keyword_chunk,
    _meta_chunk,
    _write_article,
    _cover_image,
    _content_photo_chunk,
    evaluate_article,
    replace_visual_markers,
    Research,
    ARTICLE_PROFILES,
)
from app.services.blog_visuals import render_visual, replace_visual_markers as _replace_visual_markers
from app.services.blog_content_gate import GateResult


async def dry_run_single(topic: str | None = None, force_topic: str | None = None) -> dict:
    """Run a single dry run, returning detailed diagnostics."""
    print(f"\n{'='*60}")
    print(f"DRY RUN: {topic or 'auto'}")
    print(f"{'='*60}")

    # Fetch recent posts for dedupe
    recent_posts = await _fetch_recent_posts()
    print(f"Recent posts for dedupe: {len(recent_posts)}")

    # Research and topic selection
    if force_topic:
        research = Research(
            topic=force_topic,
            keywords=[],
            snippets=["Forced topic for testing"],
            source_urls=[],
        )
        results = []
    else:
        research, results = await _research_and_select_topic(recent_posts)
        print(f"Selected topic: {research.topic}")

    # Store source URLs
    research.source_urls = [r.get("url", "") for r in results if r.get("url")]

    # Keywords + article type
    keyword_plan = await _keyword_chunk(research.topic, results)
    research.keywords = keyword_plan.keywords
    research.article_type = keyword_plan.article_type
    print(f"Keywords: {research.keywords}")
    print(f"Article type: {research.article_type}")

    # Data availability & profile
    from app.services.auto_blog import _analyze_data_availability, _build_article_profile
    data = _analyze_data_availability(research)
    profile = _build_article_profile(research.article_type, research.topic, data)
    research.data_availability = data
    research.profile = profile

    print(f"\n--- Data Availability ---")
    print(f"Numbers: {data.numbers}")
    print(f"Named items: {data.named_items}")
    print(f"Attributes: {data.attributes}")
    print(f"Chart data: {data.has_chart_data}")
    print(f"Table data: {data.has_table_data}")
    print(f"Summary: {data.summary}")

    print(f"\n--- Article Profile ---")
    print(f"Type: {profile['article_type']}")
    print(f"Cover: {profile.get('cover_photo')}")
    print(f"Max content photos: {profile.get('max_content_photos')}")
    print(f"Required visuals: {profile.get('required_visuals', ())}")
    print(f"Max visuals: {profile.get('max_visuals')}")
    if profile.get("downgraded"):
        print(f"DOWNGRADED: {profile.get('downgrade_reason')}")

    # Meta
    meta_title, meta_description = await _meta_chunk(research)
    print(f"\nMeta title: {meta_title}")
    print(f"Meta description: {meta_description}")

    # Article with visuals
    article = await _write_article(research)
    print(f"\n--- Article ---")
    print(f"Title: {article['title']}")
    print(f"Excerpt: {article['excerpt']}")
    print(f"Tags: {article['tags']}")
    print(f"Visuals raw: {len(article.get('visuals', []))}")

    # Visual validation
    visuals = article.get("visuals", [])
    profile = research.profile or {}
    required_visuals = profile.get("required_visuals", ())
    data_avail = research.data_availability

    visual_failures = []
    for i, v in enumerate(visuals):
        kind = v.get("kind", "")
        if kind == "table":
            if not v.get("columns") or not v.get("rows") or len(v.get("rows", [])) < 2 or len(v.get("columns", [])) < 3:
                visual_failures.append(f"Visual {i}: table needs ≥2 rows and ≥3 columns")
        elif kind in ("bar", "line"):
            labels = v.get("labels", [])
            values = v.get("values", [])
            if len(labels) < 3 or len(values) < 3 or len(labels) != len(values):
                visual_failures.append(f"Visual {i}: {kind} chart needs ≥3 data points")
            num_values = [_to_number(v) for v in values]
            if num_values and len(set(num_values)) == 1:
                visual_failures.append(f"Visual {i}: chart values have no variation")

    if required_visuals and data_avail:
        for req in required_visuals:
            has_req = any(v.get("kind") == req for v in visuals)
            if not has_req:
                if req in ("bar", "line") and data_avail.has_chart_data:
                    visual_failures.append(f"Required {req} chart missing despite chart data available")
                elif req == "table" and data_avail.has_table_data:
                    visual_failures.append(f"Required table missing despite table data available")

    # Content gate evaluation
    from app.services.auto_blog import _to_number
    json_meta = article.get("_json_meta", {})
    json_repaired = json_meta.get("repaired", False)

    gate_result = evaluate_article(
        article, "", "", research, profile, json_repaired
    )

    # Add visual failures to gate
    if visual_failures:
        gate_result.reasons.extend(visual_failures)
        gate_result.passed = False

    # Check required visuals missing while data exists
    if required_visuals and data_avail:
        for req in required_visuals:
            has_req = any(v.get("kind") == req for v in visuals)
            if not has_req:
                if req in ("bar", "line") and data_avail.has_chart_data:
                    visual_failures.append(f"Required {req} chart missing despite chart data available")
                elif req == "table" and data_avail.has_table_data:
                    visual_failures.append(f"Required table missing despite table data available")

    if visual_failures:
        gate_result.reasons.extend(visual_failures)
        gate_result.passed = False

    if json_repaired:
        gate_result.reasons.append("JSON was repaired from truncation")
        gate_result.passed = False

    print(f"\n--- Content Gate ---")
    print(f"Passed: {gate_result.passed}")
    print(f"Reasons: {gate_result.reasons if gate_result.reasons else 'None'}")
    print(f"Meta-only failure: {gate_result.meta_only_failure}")

    # Render visuals
    content_html = article["content_html"]
    visuals = article.get("visuals", [])
    highlight_best = profile.get("article_type") == "benchmark"
    rendered_html, dropped = replace_visual_markers(content_html, visuals, highlight_best=highlight_best)
    if dropped:
        print(f"Dropped visuals: {dropped}")

    # Images
    cover, cover_source = await _cover_image("Test title", ["test"])
    content_html_with_photo, content_images = await _content_photo_chunk(
        rendered_html, profile, ["test"], {cover, cover_source} if cover else set()
    )

    return {
        "topic": research.topic,
        "article_type": profile.get("article_type"),
        "downgraded": profile.get("downgraded", False),
        "downgrade_reason": profile.get("downgrade_reason", ""),
        "keywords": research.keywords,
        "data_availability": {
            "numbers": data_avail.numbers,
            "named_items": data_avail.named_items,
            "attributes": data_avail.attributes,
            "has_chart_data": data_avail.has_chart_data,
            "has_table_data": data_avail.has_table_data,
            "summary": data_avail.summary,
        },
        "profile": {
            "type": profile.get("article_type"),
            "cover": profile.get("cover_photo"),
            "max_content_photos": profile.get("max_content_photos"),
            "required_visuals": list(profile.get("required_visuals", ())),
            "max_visuals": profile.get("max_visuals"),
        },
        "visuals": [
            {
                "kind": v.get("kind"),
                "title": v.get("title"),
                "kept": True,
            }
            for v in visuals
        ],
        "visual_failures": visual_failures,
        "photo_count": 1 if cover else 0,
        "gate": {
            "passed": gate_result.passed,
            "reasons": gate_result.reasons,
            "meta_only_failure": gate_result.meta_only_failure,
        },
        "word_count": len(re.sub(r"<[^>]+>", " ", article["content_html"]).split()),
        "rendered_html_preview": rendered_html[:2000],
    }


def _to_number(val):
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        s = val.strip().replace(",", "").replace("%", "")
        try:
            return float(s)
        except ValueError:
            return None
    return None


import re


async def main():
    parser = argparse.ArgumentParser(description="AutoBlog dry run")
    parser.add_argument("--topic", help="Force a specific topic")
    parser.add_argument("--count", type=int, default=1, help="Number of runs")
    parser.add_argument("--output", help="Output JSON file")
    args = parser.parse_args()

    all_results = []
    for i in range(args.count):
        result = await dry_run_single(topic=args.topic)
        all_results.append(result)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")

    # Print summary
    for r in all_results:
        print(f"\n--- SUMMARY ---")
        print(f"Topic: {r['topic']}")
        print(f"Type: {r['article_type']} (downgraded: {r['downgraded']})")
        print(f"Words: {r['word_count']}")
        print(f"Gate: {'PASS' if r['gate']['passed'] else 'FAIL'}")
        if r['gate']['reasons']:
            print(f"  Failures: {r['gate']['reasons']}")
        print(f"Visuals: {len(r['visuals'])} kept, {len(r['visual_failures'])} dropped")
        if r['visual_failures']:
            for vf in r['visual_failures']:
                print(f"  - {vf}")
        print(f"Photos: {r['photo_count']}")

    # Save HTML output
    if args.output:
        html_path = Path(args.output).with_suffix(".html")
        with open(html_path, "w") as f:
            f.write("<html><body>")
            for r in all_results:
                f.write(f"<h2>{r['topic']}</h2>")
                f.write(f"<pre>{json.dumps(r, indent=2, default=str)}</pre>")
                if 'rendered_html_preview' in r:
                    f.write(r['rendered_html_preview'])
            f.write("</body></html>")
        print(f"HTML preview saved to {html_path}")


if __name__ == "__main__":
    asyncio.run(main())