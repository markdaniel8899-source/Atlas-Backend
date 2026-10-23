import io
import unittest
from unittest import mock

from PIL import Image

from app.services import r2_storage as r2_storage_module


def _png_bytes(size=(600, 600), color="red") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


class TestR2ImageStorage(unittest.TestCase):
    def test_compresses_to_webp_and_uploads(self):
        storage = r2_storage_module.R2ImageStorage()
        fake_client = mock.Mock()
        storage._client = fake_client

        with (
            mock.patch.object(
                type(storage),
                "bucket_name",
                new_callable=mock.PropertyMock,
                return_value="atlas-blog-images",
            ),
            mock.patch.object(
                type(storage),
                "public_url",
                new_callable=mock.PropertyMock,
                return_value="https://cdn.example.com",
            ),
        ):
            url = storage.compress_and_upload(
                _png_bytes(), "My Cover Slug & More"
            )

        self.assertTrue(
            url.startswith("https://cdn.example.com/covers/my-cover-slug-more_")
        )
        self.assertTrue(url.endswith(".webp"))

        kwargs = fake_client.put_object.call_args.kwargs
        self.assertEqual(kwargs["Bucket"], "atlas-blog-images")
        self.assertEqual(kwargs["ContentType"], "image/webp")
        self.assertIn("max-age=31536000", kwargs["CacheControl"])

        with Image.open(io.BytesIO(kwargs["Body"])) as image:
            self.assertEqual(image.format, "WEBP")
            self.assertEqual(image.size, (1200, 630))

    def test_missing_configuration_raises(self):
        storage = r2_storage_module.R2ImageStorage()
        empty = {
            "R2_ENDPOINT_URL": "",
            "R2_ACCESS_KEY_ID": "",
            "R2_SECRET_ACCESS_KEY": "",
            "R2_BUCKET_NAME": "",
            "R2_PUBLIC_URL": "",
        }
        with mock.patch.dict(r2_storage_module.os.environ, empty):
            with self.assertRaises(RuntimeError):
                storage.compress_and_upload(_png_bytes(), "slug")

    def test_empty_image_bytes_raises(self):
        storage = r2_storage_module.R2ImageStorage()
        with self.assertRaises(ValueError):
            storage.compress_and_upload(b"", "slug")


if __name__ == "__main__":
    unittest.main()
