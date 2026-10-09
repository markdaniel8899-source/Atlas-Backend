import asyncio
import unittest

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


class TestCoverImageUrl(unittest.TestCase):
    def test_format_matches_spec(self):
        url = auto_blog._cover_image_url("Future of AI", ["ai education"])
        self.assertTrue(url.startswith("https://image.pollinations.ai/prompt/"))
        self.assertEqual(url.split("?", 1)[1], "width=1200&height=630&nologo=true")

    def test_prompt_is_url_encoded(self):
        url = auto_blog._cover_image_url("A, B & C?", [])
        prompt = url.split("/prompt/", 1)[1].split("?", 1)[0]
        self.assertNotIn(" ", prompt)
        self.assertNotIn("&", prompt)


class TestContentImageHelpers(unittest.TestCase):
    def test_content_image_url_format(self):
        url = auto_blog._content_image_url("Focus techniques", "deep work")
        self.assertTrue(url.startswith("https://image.pollinations.ai/prompt/"))
        self.assertEqual(url.split("?", 1)[1], "width=800&height=450&nologo=true")

    def test_embeds_one_figure_per_h2(self):
        html_in = (
            "<p>Intro paragraph.</p>"
            "<h2>Section One</h2><p>Body one.</p>"
            "<h2>Section Two</h2><p>Body two.</p>"
            "<h2>Section Three</h2><p>Body three.</p>"
            "<h2>Section Four</h2><p>Body four.</p>"
        )
        out, images = auto_blog._embed_content_images(html_in, ["ai", "learning"])
        self.assertEqual(len(images), auto_blog.MAX_CONTENT_IMAGES)
        self.assertEqual(out.count("<figure>"), auto_blog.MAX_CONTENT_IMAGES)
        self.assertEqual(out.count("<figcaption>"), auto_blog.MAX_CONTENT_IMAGES)
        # Fourth section stays image-free.
        self.assertIn("<h2>Section Four</h2><p>Body four.</p>", out)
        # Figure sits at the END of its section (before the next <h2>).
        self.assertIn("</p><figure>", out)
        self.assertIn("width=800&height=450&nologo=true", out)
        self.assertEqual(images[0]["caption"], "Section One")
        self.assertIn("Section%20One", images[0]["url"])

    def test_no_h2_appends_single_image(self):
        html_in = "<p>First para.</p><p>Second para.</p>"
        out, images = auto_blog._embed_content_images(html_in, ["ai"])
        self.assertEqual(len(images), 1)
        self.assertEqual(out.count("<figure>"), 1)
        self.assertIn("</p><figure>", out)

    def test_captions_are_escaped(self):
        html_in = '<h2>Tom & Jerry\'s "Study" <Guide></h2><p>x</p>'
        out, images = auto_blog._embed_content_images(html_in, ["ai"])
        self.assertEqual(len(images), 1)
        # "<Guide>" is tag-stripped from the caption, then entities are escaped.
        self.assertIn("<figcaption>Tom &amp; Jerry&#x27;s &quot;Study&quot;</figcaption>", out)
        self.assertIn('alt="Tom &amp; Jerry&#x27;s &quot;Study&quot;"', out)
        self.assertNotIn('alt="Tom & Jerry', out)

    def test_empty_keywords_skips_images(self):
        html_in = "<h2>Only</h2><p>x</p>"
        out, images = auto_blog._embed_content_images(html_in, [])
        self.assertEqual(images, [])
        self.assertNotIn("<figure>", out)


class TestPipelineGraceful(unittest.TestCase):
    def test_run_pipeline_returns_none_without_keys(self):
        """No GOOGLE/TAVILY keys → stage error is logged, None returned."""
        result = asyncio.run(auto_blog.run_pipeline())
        self.assertIsNone(result)
        self.assertTrue(auto_blog._RUN_LOCK.acquire(blocking=False))
        auto_blog._RUN_LOCK.release()


if __name__ == "__main__":
    unittest.main()
