import asyncio
import unittest
from unittest import mock

from app.services import auto_blog


class TestSlugify(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(auto_blog.slugify("AI & Technology: 5 Trends!"), "ai-technology-5-trends")

    def test_leading_trailing_junk(self):
        self.assertEqual(auto_blog.slugify("  --Weird//Title??  "), "weird-title")

    def test_empty_falls_back(self):
        self.assertRegex(auto_blog.slugify("!!!"), r"^blog-\d{8}$")

    def test_matches_db_check_constraint(self):
        import re

        pat = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
        for title in ["GPT-4o Changes Everything", "A", "100% Free AI Tools 2026"]:
            self.assertRegex(auto_blog.slugify(title), pat)


def _fake_pixabay_client(hits):
    """Build a mock httpx.AsyncClient context manager returning `hits`."""
    response = mock.Mock()
    response.raise_for_status = mock.Mock()
    response.json.return_value = {"hits": hits}
    client = mock.MagicMock()
    client.get = mock.AsyncMock(return_value=response)
    cm = mock.MagicMock()
    cm.__aenter__ = mock.AsyncMock(return_value=client)
    cm.__aexit__ = mock.AsyncMock(return_value=False)
    return client, cm


def _resp(hits):
    response = mock.Mock()
    response.raise_for_status = mock.Mock()
    response.json.return_value = {"hits": hits}
    return response


def _client_with_responses(responses):
    client = mock.MagicMock()
    client.get = mock.AsyncMock(side_effect=responses)
    cm = mock.MagicMock()
    cm.__aenter__ = mock.AsyncMock(return_value=client)
    cm.__aexit__ = mock.AsyncMock(return_value=False)
    return client, cm


class TestFetchRelevantImage(unittest.TestCase):
    def test_missing_key_returns_none(self):
        with mock.patch.object(auto_blog, "PIXABAY_API_KEY", ""):
            result = asyncio.run(
                auto_blog.fetch_relevant_image("ai education", ["deep learning"])
            )
        self.assertIsNone(result)

    def test_title_topic_is_first_query(self):
        client, cm = _fake_pixabay_client(
            [
                {
                    "tags": "ai, software",
                    "largeImageURL": "https://cdn.pixabay.com/photo/1-large.jpg",
                }
            ]
        )
        with (
            mock.patch.object(auto_blog, "PIXABAY_API_KEY", "test-key"),
            mock.patch.object(auto_blog.httpx, "AsyncClient", return_value=cm),
        ):
            url = asyncio.run(
                auto_blog.fetch_relevant_image(
                    "AI Is Eating Software", ["ai tools"], category="technology"
                )
            )
        self.assertEqual(url, "https://cdn.pixabay.com/photo/1-large.jpg")
        params = client.get.call_args.kwargs["params"]
        self.assertEqual(params["q"], "AI Is Eating Software")
        self.assertEqual(params["image_type"], "photo")
        self.assertEqual(params["orientation"], "horizontal")
        self.assertEqual(params["per_page"], 5)
        self.assertEqual(params["min_width"], 1200)
        self.assertEqual(params["order"], "popular")
        # Tech queries send a valid Pixabay category, not "technology".
        self.assertEqual(params["category"], "computer")

    def test_model_name_is_first_query(self):
        client, cm = _fake_pixabay_client(
            [
                {
                    "tags": "ai, computer",
                    "largeImageURL": "https://cdn.pixabay.com/gpt.jpg",
                }
            ]
        )
        with (
            mock.patch.object(auto_blog, "PIXABAY_API_KEY", "test-key"),
            mock.patch.object(auto_blog.httpx, "AsyncClient", return_value=cm),
        ):
            url = asyncio.run(
                auto_blog.fetch_relevant_image(
                    "OpenAI launches GPT-5.5 with new coding tools",
                    ["gpt-5.5", "openai"],
                )
            )
        self.assertEqual(url, "https://cdn.pixabay.com/gpt.jpg")
        params = client.get.call_args.kwargs["params"]
        self.assertEqual(params["q"], "GPT-5.5 AI")
        self.assertEqual(params["category"], "computer")

    def test_fintech_query_uses_business_category(self):
        client, cm = _fake_pixabay_client(
            [
                {
                    "tags": "finance, technology",
                    "largeImageURL": "https://cdn.pixabay.com/fintech.jpg",
                }
            ]
        )
        with (
            mock.patch.object(auto_blog, "PIXABAY_API_KEY", "test-key"),
            mock.patch.object(auto_blog.httpx, "AsyncClient", return_value=cm),
        ):
            asyncio.run(
                auto_blog.fetch_relevant_image(
                    "AI fintech regulations update", ["fintech", "banking"]
                )
            )
        params = client.get.call_args.kwargs["params"]
        self.assertEqual(params["category"], "business")

    def test_non_tech_query_does_not_send_category(self):
        client, cm = _fake_pixabay_client(
            [
                {
                    "tags": "cake, food",
                    "largeImageURL": "https://cdn.pixabay.com/cake.jpg",
                }
            ]
        )
        with (
            mock.patch.object(auto_blog, "PIXABAY_API_KEY", "test-key"),
            mock.patch.object(auto_blog.httpx, "AsyncClient", return_value=cm),
        ):
            asyncio.run(
                auto_blog.fetch_relevant_image(
                    "Birthday Cake Decorating Tips", ["baking"]
                )
            )
        params = client.get.call_args.kwargs["params"]
        self.assertNotIn("category", params)
        self.assertEqual(params["q"], "Birthday Cake Decorating Tips")

    def test_tech_tags_beat_generic_business_hits(self):
        business = {
            "tags": "business, office, meeting",
            "largeImageURL": "https://cdn.pixabay.com/business.jpg",
        }
        coding = {
            "tags": "coding, software, computer",
            "largeImageURL": "https://cdn.pixabay.com/coding.jpg",
        }
        _, cm = _fake_pixabay_client([business, coding])
        with (
            mock.patch.object(auto_blog, "PIXABAY_API_KEY", "test-key"),
            mock.patch.object(auto_blog.httpx, "AsyncClient", return_value=cm),
        ):
            url = asyncio.run(
                auto_blog.fetch_relevant_image("AI agents for coding", ["coding"])
            )
        self.assertEqual(url, "https://cdn.pixabay.com/coding.jpg")

    def test_second_query_used_when_first_empty(self):
        client, cm = _client_with_responses(
            [_resp([]), _resp([{"tags": "ai", "largeImageURL": "https://cdn.pixabay.com/2.jpg"}])]
        )
        with (
            mock.patch.object(auto_blog, "PIXABAY_API_KEY", "test-key"),
            mock.patch.object(auto_blog.httpx, "AsyncClient", return_value=cm),
        ):
            url = asyncio.run(
                auto_blog.fetch_relevant_image("AI Is Eating Software", ["ai tools"])
            )
        self.assertEqual(url, "https://cdn.pixabay.com/2.jpg")
        self.assertEqual(client.get.call_count, 2)
        # Second query = first two keywords.
        self.assertEqual(client.get.call_args_list[1].kwargs["params"]["q"], "ai tools")

    def test_food_decoy_hits_are_skipped(self):
        cake = {
            "tags": "cake, food, eating",
            "largeImageURL": "https://cdn.pixabay.com/cake.jpg",
        }
        tech = {
            "tags": "ai, software",
            "largeImageURL": "https://cdn.pixabay.com/tech.jpg",
        }
        _, cm = _fake_pixabay_client([cake, tech])
        with (
            mock.patch.object(auto_blog, "PIXABAY_API_KEY", "test-key"),
            mock.patch.object(auto_blog.httpx, "AsyncClient", return_value=cm),
        ):
            url = asyncio.run(
                auto_blog.fetch_relevant_image("AI Is Eating Software", ["ai"])
            )
        self.assertEqual(url, "https://cdn.pixabay.com/tech.jpg")

    def test_genuinely_foody_query_keeps_food_hits(self):
        cake = {
            "tags": "cake, food, birthday",
            "largeImageURL": "https://cdn.pixabay.com/cake.jpg",
        }
        _, cm = _fake_pixabay_client([cake])
        with (
            mock.patch.object(auto_blog, "PIXABAY_API_KEY", "test-key"),
            mock.patch.object(auto_blog.httpx, "AsyncClient", return_value=cm),
        ):
            url = asyncio.run(
                auto_blog.fetch_relevant_image("Birthday Cake Decorating Tips", ["baking"])
            )
        self.assertEqual(url, "https://cdn.pixabay.com/cake.jpg")

    def test_http_errors_on_all_queries_return_unsplash_fallback(self):
        client, cm = _fake_pixabay_client([])
        client.get.side_effect = RuntimeError("boom")
        with (
            mock.patch.object(auto_blog, "PIXABAY_API_KEY", "test-key"),
            mock.patch.object(auto_blog.httpx, "AsyncClient", return_value=cm),
        ):
            url = asyncio.run(auto_blog.fetch_relevant_image("ai", ["ai"]))
        self.assertEqual(url, auto_blog.UNSPLASH_FALLBACK_IMAGE)

    def test_ladder_exhausted_returns_unsplash_fallback(self):
        _, cm = _fake_pixabay_client([])
        with (
            mock.patch.object(auto_blog, "PIXABAY_API_KEY", "test-key"),
            mock.patch.object(auto_blog.httpx, "AsyncClient", return_value=cm),
        ):
            url = asyncio.run(
                auto_blog.fetch_relevant_image("Totally Unknown Thing", ["zzz"])
            )
        self.assertEqual(url, auto_blog.UNSPLASH_FALLBACK_IMAGE)


class TestResearchAndDedupe(unittest.TestCase):
    def test_queries_rotate_daily_and_use_current_month(self):
        with (
            mock.patch.object(auto_blog, "_day_index", return_value=1),
            mock.patch.object(auto_blog, "_month_year", return_value="October 2026"),
        ):
            queries = auto_blog._ordered_research_queries()
        self.assertEqual(queries[0][0], "industry_news")
        self.assertEqual(queries[-1][0], "model_releases")
        rendered = " ".join(query for _, query in queries)
        self.assertIn("October 2026", rendered)

    def test_tavily_uses_news_mode_and_official_domains(self):
        client = mock.Mock()
        client.search.return_value = {"results": []}
        settings = mock.Mock(tavily_api_key="test-key")
        with (
            mock.patch.object(auto_blog, "get_settings", return_value=settings),
            mock.patch.object(auto_blog, "TavilyClient", return_value=client),
        ):
            auto_blog._tavily_search("latest AI news")
        kwargs = client.search.call_args.kwargs
        self.assertEqual(kwargs["topic"], "news")
        self.assertEqual(kwargs["days"], 3)
        self.assertEqual(kwargs["search_depth"], "advanced")
        self.assertEqual(kwargs["max_results"], 10)
        self.assertIn("openai.com", kwargs["include_domains"])
        self.assertNotIn(
            "techcrunch.com/category/artificial-intelligence",
            kwargs["include_domains"],
        )

    def test_similarity_flags_repeated_angles(self):
        score = auto_blog._topic_similarity(
            "GPT-5.5 launch brings coding upgrades",
            "GPT-5.5 release adds new coding upgrades",
        )
        self.assertGreater(score, auto_blog.SIMILARITY_THRESHOLD)
        different = auto_blog._topic_similarity(
            "AI tutoring tools for students",
            "AI fintech regulations update",
        )
        self.assertLessEqual(different, auto_blog.SIMILARITY_THRESHOLD)

    def test_find_duplicate_checks_keywords_too(self):
        score, match = auto_blog._find_duplicate(
            "Anthropic IPO",
            [{"title": "A different headline", "keywords": ["anthropic ipo"]}],
        )
        self.assertGreater(score, auto_blog.SIMILARITY_THRESHOLD)
        self.assertEqual(match, "anthropic ipo")

    def test_retry_rejects_duplicate_then_accepts_new_topic(self):
        results = [
            {
                "title": "GPT-5.5 release",
                "url": "https://openai.com/index/gpt-5-5",
                "content": "OpenAI released GPT-5.5.",
                "published_date": "2026-10-23",
            }
        ]
        recent = [{"title": "GPT-5.5 Released by OpenAI", "keywords": ["gpt-5.5"]}]
        ordered = [
            ("model_releases", "q1"),
            ("industry_news", "q2"),
            ("research", "q3"),
            ("education", "q4"),
            ("fintech", "q5"),
        ]
        topic_mock = mock.AsyncMock(
            side_effect=[
                "GPT-5.5 Released by OpenAI",
                "AI tutoring tools for students",
            ]
        )
        with (
            mock.patch.object(auto_blog, "_ordered_research_queries", return_value=ordered),
            mock.patch.object(auto_blog, "_research_chunk", mock.AsyncMock(return_value=results)),
            mock.patch.object(auto_blog, "_topic_chunk", topic_mock),
        ):
            research, selected = asyncio.run(
                auto_blog._research_and_select_topic(recent)
            )
        self.assertEqual(research.topic, "AI tutoring tools for students")
        self.assertEqual(selected, results)
        self.assertEqual(topic_mock.await_count, 2)
        self.assertEqual(topic_mock.await_args_list[0].kwargs["category"], "model_releases")
        self.assertEqual(topic_mock.await_args_list[1].kwargs["category"], "industry_news")

    def test_forces_unused_category_after_two_retries(self):
        results = [
            {
                "title": "GPT-5.5 release",
                "url": "https://openai.com/index/gpt-5-5",
                "content": "OpenAI released GPT-5.5.",
                "published_date": "2026-10-23",
            }
        ]
        recent = [{"title": "GPT-5.5 Released by OpenAI", "keywords": ["gpt-5.5"]}]
        ordered = [
            ("model_releases", "q1"),
            ("industry_news", "q2"),
            ("research", "q3"),
            ("education", "q4"),
            ("fintech", "q5"),
        ]
        topic_mock = mock.AsyncMock(
            side_effect=[
                "GPT-5.5 Released by OpenAI",
                "GPT-5.5 Released by OpenAI",
                "GPT-5.5 Released by OpenAI",
                "AI tutoring tools for students",
            ]
        )
        with (
            mock.patch.object(auto_blog, "_ordered_research_queries", return_value=ordered),
            mock.patch.object(auto_blog, "_research_chunk", mock.AsyncMock(return_value=results)),
            mock.patch.object(auto_blog, "_topic_chunk", topic_mock),
        ):
            research, _ = asyncio.run(
                auto_blog._research_and_select_topic(recent)
            )
        self.assertEqual(research.topic, "AI tutoring tools for students")
        self.assertEqual(topic_mock.await_count, 4)
        self.assertEqual(topic_mock.await_args_list[-1].kwargs["category"], "education")

    def test_fetch_recent_posts_queries_latest_ten(self):
        table = mock.Mock()
        query = table.select.return_value.order.return_value.limit.return_value
        query.execute.return_value = mock.Mock(
            data=[{"title": "Recent post", "keywords": ["ai"], "published_at": "x"}]
        )
        client = mock.Mock()
        client.table.return_value = table
        with mock.patch.object(auto_blog, "_supabase", return_value=client):
            rows = auto_blog._fetch_recent_posts(7)
        self.assertEqual(rows[0]["title"], "Recent post")
        table.select.assert_called_once_with("title, keywords, published_at")
        table.select.return_value.order.assert_called_once_with(
            "published_at", desc=True
        )
        table.select.return_value.order.return_value.limit.assert_called_once_with(7)

    def test_fetch_recent_posts_failure_is_non_fatal(self):
        with mock.patch.object(
            auto_blog, "_supabase", side_effect=RuntimeError("db down")
        ):
            self.assertEqual(auto_blog._fetch_recent_posts(), [])

    def test_blocked_topics_include_titles_and_keywords(self):
        blocked = auto_blog._blocked_topics(
            [
                {
                    "title": "GPT-5.5 Released by OpenAI",
                    "keywords": ["gpt-5.5", "openai", "ai"],
                }
            ]
        )
        self.assertEqual(
            blocked,
            ["GPT-5.5 Released by OpenAI", "gpt-5.5", "openai"],
        )


def _fake_download_client(content=b"fake-image", content_type="image/jpeg"):
    response = mock.Mock()
    response.raise_for_status = mock.Mock()
    response.headers = {"content-type": content_type}
    response.content = content
    client = mock.MagicMock()
    client.get = mock.AsyncMock(return_value=response)
    cm = mock.MagicMock()
    cm.__aenter__ = mock.AsyncMock(return_value=client)
    cm.__aexit__ = mock.AsyncMock(return_value=False)
    return client, cm


class TestCoverImageUrl(unittest.TestCase):
    def test_pixabay_hit_is_persisted_to_r2(self):
        with (
            mock.patch.object(
                auto_blog,
                "fetch_relevant_image",
                mock.AsyncMock(
                    return_value="https://pixabay.com/get/expiring-cover.jpg"
                ),
            ),
            mock.patch.object(
                auto_blog,
                "_persist_cover_image",
                mock.AsyncMock(
                    return_value="https://cdn.example.com/covers/future-of-ai.webp"
                ),
            ) as persist,
        ):
            url = asyncio.run(
                auto_blog._cover_image_url("Future of AI", ["ai education"])
            )

        self.assertEqual(
            url, "https://cdn.example.com/covers/future-of-ai.webp"
        )
        persist.assert_awaited_once_with(
            "https://pixabay.com/get/expiring-cover.jpg", "future-of-ai"
        )

    def test_missing_candidate_returns_stable_unsplash(self):
        with mock.patch.object(
            auto_blog, "fetch_relevant_image", mock.AsyncMock(return_value=None)
        ):
            url = asyncio.run(
                auto_blog._cover_image_url("Future of AI", ["ai education"])
            )
        self.assertEqual(url, auto_blog.UNSPLASH_FALLBACK_IMAGE)

    def test_persist_downloads_and_uploads_cover(self):
        _, cm = _fake_download_client()
        storage = mock.Mock()
        storage.compress_and_upload.return_value = (
            "https://cdn.example.com/covers/ai_1234.webp"
        )
        with (
            mock.patch.object(auto_blog.httpx, "AsyncClient", return_value=cm),
            mock.patch.object(auto_blog, "r2_storage", storage),
        ):
            url = asyncio.run(
                auto_blog._persist_cover_image(
                    "https://pixabay.com/get/expiring.jpg", "ai"
                )
            )

        self.assertEqual(url, "https://cdn.example.com/covers/ai_1234.webp")
        storage.compress_and_upload.assert_called_once_with(b"fake-image", "ai")

    def test_upload_failure_returns_stable_unsplash(self):
        _, cm = _fake_download_client()
        storage = mock.Mock()
        storage.compress_and_upload.side_effect = RuntimeError("R2 down")
        with (
            mock.patch.object(auto_blog.httpx, "AsyncClient", return_value=cm),
            mock.patch.object(auto_blog, "r2_storage", storage),
        ):
            url = asyncio.run(
                auto_blog._persist_cover_image(
                    "https://pixabay.com/get/expiring.jpg", "ai"
                )
            )

        self.assertEqual(url, auto_blog.UNSPLASH_FALLBACK_IMAGE)

    def test_non_image_download_returns_stable_unsplash(self):
        _, cm = _fake_download_client(content=b"<html>", content_type="text/html")
        with mock.patch.object(auto_blog.httpx, "AsyncClient", return_value=cm):
            url = asyncio.run(
                auto_blog._persist_cover_image(
                    "https://pixabay.com/get/expiring.jpg", "ai"
                )
            )

        self.assertEqual(url, auto_blog.UNSPLASH_FALLBACK_IMAGE)


class TestContentImageHelpers(unittest.TestCase):
    def test_missing_pixabay_key_uses_unsplash_fallback(self):
        with mock.patch.object(
            auto_blog, "fetch_relevant_image", mock.AsyncMock(return_value=None)
        ):
            url = asyncio.run(
                auto_blog._content_image_url("Focus techniques", "deep work")
            )
        self.assertEqual(url, auto_blog.UNSPLASH_FALLBACK_IMAGE)

    def test_excluded_fallback_moves_to_next_unsplash(self):
        with mock.patch.object(
            auto_blog, "fetch_relevant_image", mock.AsyncMock(return_value=None)
        ):
            url = asyncio.run(
                auto_blog._content_image_url(
                    "Focus techniques",
                    "deep work",
                    exclude_urls={auto_blog.UNSPLASH_FALLBACK_IMAGE},
                )
            )
        self.assertEqual(url, auto_blog.UNSPLASH_FALLBACK_IMAGES[1])

    def test_pixabay_hit_wins(self):
        with mock.patch.object(
            auto_blog,
            "fetch_relevant_image",
            mock.AsyncMock(return_value="https://cdn.pixabay.com/inline.jpg"),
        ):
            url = asyncio.run(
                auto_blog._content_image_url("Focus techniques", "deep work")
            )
        self.assertEqual(url, "https://cdn.pixabay.com/inline.jpg")

    def test_exclusions_are_forwarded_to_fetch(self):
        fetch = mock.AsyncMock(return_value="https://cdn.pixabay.com/inline.jpg")
        with mock.patch.object(auto_blog, "fetch_relevant_image", fetch):
            asyncio.run(
                auto_blog._content_image_url(
                    "Focus techniques",
                    "deep work",
                    exclude_urls={"https://cdn.pixabay.com/cover.jpg"},
                )
            )
        self.assertEqual(
            fetch.call_args.kwargs["exclude_urls"],
            {"https://cdn.pixabay.com/cover.jpg"},
        )


class TestPickHitExclusions(unittest.TestCase):
    def test_skips_excluded_urls(self):
        hits = [
            {"largeImageURL": "https://cdn.pixabay.com/cover.jpg", "tags": "coding"},
            {"largeImageURL": "https://cdn.pixabay.com/second.jpg", "tags": "coding"},
        ]
        picked = auto_blog._pick_hit(
            hits, "ai coding", {"https://cdn.pixabay.com/cover.jpg"}
        )
        self.assertEqual(picked, "https://cdn.pixabay.com/second.jpg")

    def test_returns_empty_when_every_hit_is_excluded(self):
        hits = [
            {"largeImageURL": "https://cdn.pixabay.com/a.jpg", "tags": "coding"},
            {"largeImageURL": "https://cdn.pixabay.com/b.jpg", "tags": "coding"},
        ]
        picked = auto_blog._pick_hit(
            hits,
            "ai coding",
            {
                "https://cdn.pixabay.com/a.jpg",
                "https://cdn.pixabay.com/b.jpg",
            },
        )
        self.assertEqual(picked, "")


def _run_images_chunk(
    html_in,
    keywords,
    image_url=None,
    *,
    image_urls=None,
    max_images=auto_blog.MAX_CONTENT_IMAGES,
    exclude_urls=None,
):
    fetch = (
        mock.AsyncMock(return_value=image_url)
        if image_urls is None
        else mock.AsyncMock(side_effect=image_urls)
    )
    with (
        mock.patch.object(auto_blog, "IMAGE_PAUSE_SECONDS", 0),
        mock.patch.object(auto_blog, "fetch_relevant_image", fetch),
    ):
        return asyncio.run(
            auto_blog._images_chunk(
                html_in,
                keywords,
                max_images=max_images,
                exclude_urls=exclude_urls,
            )
        )


class TestImagesChunk(unittest.TestCase):
    def test_embeds_one_figure_per_h2(self):
        html_in = (
            "<p>Intro paragraph.</p>"
            "<h2>Section One</h2><p>Body one.</p>"
            "<h2>Section Two</h2><p>Body two.</p>"
            "<h2>Section Three</h2><p>Body three.</p>"
            "<h2>Section Four</h2><p>Body four.</p>"
        )
        out, images = _run_images_chunk(html_in, ["ai", "learning"])
        self.assertEqual(len(images), auto_blog.MAX_CONTENT_IMAGES)
        # Plain <img> tags — no <figure> wrapper box, no captions rendered.
        self.assertEqual(out.count("<img "), auto_blog.MAX_CONTENT_IMAGES)
        self.assertNotIn("<figcaption>", out)
        self.assertNotIn("<figure>", out)
        # Fourth section stays image-free.
        self.assertIn("<h2>Section Four</h2><p>Body four.</p>", out)
        # Image sits at the END of its section (before the next <h2>).
        self.assertIn("</p><img", out)
        self.assertEqual(images[0]["caption"], "Section One")

    def test_uses_pixabay_url_when_available(self):
        html_in = (
            "<h2>Section One</h2><p>Body one.</p>"
            "<h2>Section Two</h2><p>Body two.</p>"
        )
        out, images = _run_images_chunk(
            html_in,
            ["ai"],
            image_urls=[
                "https://cdn.pixabay.com/x.jpg",
                "https://cdn.pixabay.com/y.jpg",
            ],
        )
        self.assertEqual(len(images), 2)
        self.assertEqual(images[0]["url"], "https://cdn.pixabay.com/x.jpg")
        self.assertEqual(images[1]["url"], "https://cdn.pixabay.com/y.jpg")
        self.assertIn('src="https://cdn.pixabay.com/x.jpg"', out)

    def test_random_target_controls_image_count(self):
        html_in = (
            "<h2>Section One</h2><p>Body one.</p>"
            "<h2>Section Two</h2><p>Body two.</p>"
            "<h2>Section Three</h2><p>Body three.</p>"
        )
        with mock.patch.object(
            auto_blog.random, "randint", return_value=1
        ) as randint:
            out, images = _run_images_chunk(
                html_in, ["ai"], max_images=None
            )
        self.assertEqual(len(images), 1)
        self.assertEqual(out.count("<img "), 1)
        randint.assert_called_once_with(
            auto_blog.MIN_CONTENT_IMAGES, auto_blog.MAX_CONTENT_IMAGES
        )

    def test_cover_and_used_urls_are_never_reused(self):
        html_in = (
            "<h2>Section One</h2><p>Body one.</p>"
            "<h2>Section Two</h2><p>Body two.</p>"
            "<h2>Section Three</h2><p>Body three.</p>"
        )
        cover = "https://cdn.pixabay.com/cover.jpg"
        out, images = _run_images_chunk(
            html_in,
            ["ai"],
            image_urls=[
                cover,
                "https://cdn.pixabay.com/second.jpg",
                "https://cdn.pixabay.com/third.jpg",
            ],
            exclude_urls={cover},
        )
        self.assertEqual(
            [image["url"] for image in images],
            [
                "https://cdn.pixabay.com/second.jpg",
                "https://cdn.pixabay.com/third.jpg",
            ],
        )
        self.assertNotIn(f'src="{cover}"', out)

    def test_falls_back_to_unsplash_per_image(self):
        html_in = (
            "<h2>Section One</h2><p>Body one.</p>"
            "<h2>Section Two</h2><p>Body two.</p>"
        )
        out, images = _run_images_chunk(html_in, ["ai"], image_url=None)
        self.assertEqual(len(images), 2)
        self.assertEqual(images[0]["url"], auto_blog.UNSPLASH_FALLBACK_IMAGE)
        self.assertIn(auto_blog.UNSPLASH_FALLBACK_IMAGE, out)

    def test_no_h2_appends_single_image(self):
        html_in = "<p>First para.</p><p>Second para.</p>"
        out, images = _run_images_chunk(html_in, ["ai"])
        self.assertEqual(len(images), 1)
        self.assertEqual(out.count("<img "), 1)
        self.assertIn("</p><img", out)

    def test_captions_are_escaped(self):
        html_in = '<h2>Tom & Jerry\'s "Study" <Guide></h2><p>x</p>'
        out, images = _run_images_chunk(html_in, ["ai"])
        self.assertEqual(len(images), 1)
        # Caption text lives in metadata + alt only; nothing under the image.
        self.assertNotIn("<figcaption>", out)
        self.assertIn('alt="Tom &amp; Jerry&#x27;s &quot;Study&quot;"', out)
        self.assertNotIn('alt="Tom & Jerry', out)
        self.assertEqual(images[0]["caption"], 'Tom & Jerry\'s "Study"')

    def test_faq_section_is_not_given_an_image(self):
        html_in = (
            "<p>Intro paragraph.</p>"
            "<h2>Section One</h2><p>Body one.</p>"
            "<h2>FAQ</h2><h3>What is it?</h3><p>A direct answer.</p>"
        )
        out, images = _run_images_chunk(html_in, ["ai"])
        self.assertEqual(len(images), 1)
        # FAQ Q&A markup stays intact, with no <img> before the next heading.
        self.assertIn(
            "<h2>FAQ</h2><h3>What is it?</h3><p>A direct answer.</p>", out
        )

    def test_empty_keywords_skips_images(self):
        html_in = "<h2>Only</h2><p>x</p>"
        out, images = _run_images_chunk(html_in, [])
        self.assertEqual(images, [])
        self.assertNotIn("<figure>", out)


class TestPipelineGraceful(unittest.TestCase):
    def test_run_pipeline_returns_none_without_keys(self):
        """Stage errors are logged, the lock is released, and None is returned."""
        with (
            mock.patch.object(auto_blog, "RESEARCH_PAUSE_SECONDS", 0),
            mock.patch.object(auto_blog, "_fetch_recent_posts", return_value=[]),
            mock.patch.object(
                auto_blog,
                "_research_and_select_topic",
                side_effect=RuntimeError("test: research unavailable"),
            ),
            mock.patch.object(
                auto_blog,
                "_keyword_chunk",
                side_effect=RuntimeError("test: keywords unavailable"),
            ),
        ):
            result = asyncio.run(auto_blog.run_pipeline())
        self.assertIsNone(result)
        self.assertTrue(auto_blog._RUN_LOCK.acquire(blocking=False))
        auto_blog._RUN_LOCK.release()


class TestStaggeredPipeline(unittest.TestCase):
    def test_skips_when_already_running(self):
        """Lock held → staggered run logs and returns None without touching APIs."""
        self.assertTrue(auto_blog._RUN_LOCK.acquire(blocking=False))
        try:
            self.assertTrue(auto_blog.is_run_active())
            result = asyncio.run(auto_blog.run_staggered_blog_pipeline())
            self.assertIsNone(result)
            self.assertTrue(auto_blog._RUN_LOCK.locked())
        finally:
            auto_blog._RUN_LOCK.release()
        self.assertFalse(auto_blog.is_run_active())


class TestNimChat(unittest.TestCase):
    def test_missing_key_raises_503(self):
        with mock.patch.object(auto_blog, "NVIDIA_API_KEY", ""):
            with self.assertRaises(auto_blog.BlogPipelineError) as ctx:
                auto_blog._nim_chat("sys", "user", temperature=0.1, max_tokens=10)
        self.assertEqual(ctx.exception.status_code, 503)

    def test_success_returns_stripped_content(self):
        fake = mock.Mock()
        fake.status_code = 200
        fake.raise_for_status = mock.Mock()
        fake.json.return_value = {"choices": [{"message": {"content": "  hello "}}]}
        with (
            mock.patch.object(auto_blog, "NVIDIA_API_KEY", "test-key"),
            mock.patch.object(auto_blog.requests, "post", return_value=fake) as post,
        ):
            text = auto_blog._nim_chat("sys", "user", temperature=0.3, max_tokens=64)
        self.assertEqual(text, "hello")
        _, kwargs = post.call_args
        self.assertTrue(
            kwargs["url"].endswith("/chat/completions")
            if "url" in kwargs
            else post.call_args[0][0].endswith("/chat/completions")
        )
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(kwargs["json"]["model"], auto_blog.NVIDIA_MODEL)
        self.assertEqual(
            kwargs["json"]["messages"],
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "user"},
            ],
        )

    def test_http_error_becomes_pipeline_error(self):
        fake = mock.Mock()
        fake.status_code = 400
        fake.raise_for_status.side_effect = RuntimeError("429 too many requests")
        with (
            mock.patch.object(auto_blog, "NVIDIA_API_KEY", "test-key"),
            mock.patch.object(auto_blog.requests, "post", return_value=fake),
        ):
            with self.assertRaises(auto_blog.BlogPipelineError):
                auto_blog._nim_chat("s", "p", temperature=0.1, max_tokens=10)

    def test_empty_text_raises_pipeline_error(self):
        fake = mock.Mock()
        fake.status_code = 200
        fake.raise_for_status = mock.Mock()
        fake.json.return_value = {"choices": [{"message": {"content": "   "}}]}
        with (
            mock.patch.object(auto_blog, "NVIDIA_API_KEY", "test-key"),
            mock.patch.object(auto_blog.requests, "post", return_value=fake),
        ):
            with self.assertRaises(auto_blog.BlogPipelineError):
                auto_blog._nim_chat("s", "p", temperature=0.1, max_tokens=10)


if __name__ == "__main__":
    unittest.main()
