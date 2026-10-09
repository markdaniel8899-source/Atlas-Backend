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
        self.assertEqual(params["q"], "AI Is Eating Software technology")
        self.assertEqual(params["image_type"], "photo")
        self.assertEqual(params["orientation"], "horizontal")
        self.assertEqual(params["per_page"], 5)
        self.assertEqual(params["min_width"], 1200)
        self.assertEqual(params["order"], "popular")
        # category is search text only — never an API param (invalid values 400).
        self.assertNotIn("category", params)

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


class TestCoverImageUrl(unittest.TestCase):
    def test_pixabay_hit_wins(self):
        with mock.patch.object(
            auto_blog,
            "fetch_relevant_image",
            mock.AsyncMock(return_value="https://cdn.pixabay.com/cover.jpg"),
        ):
            url = asyncio.run(
                auto_blog._cover_image_url("Future of AI", ["ai education"])
            )
        self.assertEqual(url, "https://cdn.pixabay.com/cover.jpg")

    def test_falls_back_to_pollinations(self):
        with mock.patch.object(
            auto_blog, "fetch_relevant_image", mock.AsyncMock(return_value=None)
        ):
            url = asyncio.run(
                auto_blog._cover_image_url("Future of AI", ["ai education"])
            )
        self.assertTrue(url.startswith("https://image.pollinations.ai/prompt/"))
        self.assertEqual(url.split("?", 1)[1], "width=1200&height=630&nologo=true")

    def test_fallback_prompt_is_url_encoded(self):
        with mock.patch.object(
            auto_blog, "fetch_relevant_image", mock.AsyncMock(return_value=None)
        ):
            url = asyncio.run(auto_blog._cover_image_url("A, B & C?", []))
        prompt = url.split("/prompt/", 1)[1].split("?", 1)[0]
        self.assertNotIn(" ", prompt)
        self.assertNotIn("&", prompt)


class TestContentImageHelpers(unittest.TestCase):
    def test_content_image_url_format(self):
        url = auto_blog._pollinations_content_url("Focus techniques", "deep work")
        self.assertTrue(url.startswith("https://image.pollinations.ai/prompt/"))
        self.assertEqual(url.split("?", 1)[1], "width=800&height=450&nologo=true")

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


def _run_images_chunk(html_in, keywords, image_url=None):
    with (
        mock.patch.object(auto_blog, "IMAGE_PAUSE_SECONDS", 0),
        mock.patch.object(
            auto_blog, "fetch_relevant_image", mock.AsyncMock(return_value=image_url)
        ),
    ):
        return asyncio.run(auto_blog._images_chunk(html_in, keywords))


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
        self.assertEqual(out.count("<figure>"), auto_blog.MAX_CONTENT_IMAGES)
        # Captions are NOT rendered under images — only the <img> itself.
        self.assertNotIn("<figcaption>", out)
        # Fourth section stays image-free.
        self.assertIn("<h2>Section Four</h2><p>Body four.</p>", out)
        # Figure sits at the END of its section (before the next <h2>).
        self.assertIn("</p><figure>", out)
        self.assertEqual(images[0]["caption"], "Section One")

    def test_uses_pixabay_url_when_available(self):
        html_in = (
            "<h2>Section One</h2><p>Body one.</p>"
            "<h2>Section Two</h2><p>Body two.</p>"
        )
        out, images = _run_images_chunk(
            html_in, ["ai"], image_url="https://cdn.pixabay.com/x.jpg"
        )
        self.assertEqual(len(images), 2)
        self.assertEqual(images[0]["url"], "https://cdn.pixabay.com/x.jpg")
        self.assertIn('src="https://cdn.pixabay.com/x.jpg"', out)

    def test_falls_back_to_pollinations_per_image(self):
        html_in = (
            "<h2>Section One</h2><p>Body one.</p>"
            "<h2>Section Two</h2><p>Body two.</p>"
        )
        out, images = _run_images_chunk(html_in, ["ai"], image_url=None)
        self.assertEqual(len(images), 2)
        self.assertIn("width=800&height=450&nologo=true", out)
        self.assertIn("Section%20One", images[0]["url"])

    def test_no_h2_appends_single_image(self):
        html_in = "<p>First para.</p><p>Second para.</p>"
        out, images = _run_images_chunk(html_in, ["ai"])
        self.assertEqual(len(images), 1)
        self.assertEqual(out.count("<figure>"), 1)
        self.assertIn("</p><figure>", out)

    def test_captions_are_escaped(self):
        html_in = '<h2>Tom & Jerry\'s "Study" <Guide></h2><p>x</p>'
        out, images = _run_images_chunk(html_in, ["ai"])
        self.assertEqual(len(images), 1)
        # Caption text lives in metadata + alt only; nothing under the image.
        self.assertNotIn("<figcaption>", out)
        self.assertIn('alt="Tom &amp; Jerry&#x27;s &quot;Study&quot;"', out)
        self.assertNotIn('alt="Tom & Jerry', out)
        self.assertEqual(images[0]["caption"], 'Tom & Jerry\'s "Study"')

    def test_empty_keywords_skips_images(self):
        html_in = "<h2>Only</h2><p>x</p>"
        out, images = _run_images_chunk(html_in, [])
        self.assertEqual(images, [])
        self.assertNotIn("<figure>", out)


class TestPipelineGraceful(unittest.TestCase):
    def test_run_pipeline_returns_none_without_keys(self):
        """No GOOGLE/TAVILY keys → stage error is logged, None returned."""
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
