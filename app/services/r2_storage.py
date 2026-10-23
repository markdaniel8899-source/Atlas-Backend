"""Cloudflare R2 storage for durable blog cover images.

Pixabay's API returns temporary ``/get/...`` URLs. Covers must be downloaded
and uploaded to R2 before they are saved to Supabase; in-article images keep
their existing external URLs.
"""

from __future__ import annotations

import io
import os
import re
import uuid

import boto3
from PIL import Image, ImageOps

COVER_SIZE = (1200, 630)
WEBP_QUALITY = 75
CACHE_CONTROL = "public, max-age=31536000, immutable"


class R2ImageStorage:
    """Upload compressed WebP covers to a public Cloudflare R2 bucket."""

    def __init__(self) -> None:
        self._client = None

    @property
    def bucket_name(self) -> str:
        return os.getenv("R2_BUCKET_NAME", "atlas-blog-images").strip()

    @property
    def public_url(self) -> str:
        return os.getenv("R2_PUBLIC_URL", "").strip().rstrip("/")

    @property
    def client(self):
        if self._client is not None:
            return self._client

        endpoint_url = os.getenv("R2_ENDPOINT_URL", "").strip()
        access_key_id = os.getenv("R2_ACCESS_KEY_ID", "").strip()
        secret_access_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
        missing = [
            name
            for name, value in (
                ("R2_ENDPOINT_URL", endpoint_url),
                ("R2_ACCESS_KEY_ID", access_key_id),
                ("R2_SECRET_ACCESS_KEY", secret_access_key),
                ("R2_BUCKET_NAME", self.bucket_name),
                ("R2_PUBLIC_URL", self.public_url),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                "R2 image storage is not configured; missing "
                + ", ".join(missing)
            )

        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name="auto",
        )
        return self._client

    def compress_and_upload(self, image_bytes: bytes, slug: str) -> str:
        """Crop/resize to 1200x630, encode WebP, upload, and return public URL."""
        if not image_bytes:
            raise ValueError("Cannot upload an empty cover image.")

        safe_slug = (
            re.sub(r"[^a-z0-9-]+", "-", (slug or "").lower()).strip("-")[:80]
            or "cover"
        )
        filename = f"covers/{safe_slug}_{uuid.uuid4().hex[:8]}.webp"

        try:
            with Image.open(io.BytesIO(image_bytes)) as source:
                image = ImageOps.exif_transpose(source)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image = ImageOps.fit(
                    image,
                    COVER_SIZE,
                    method=Image.Resampling.LANCZOS,
                    centering=(0.5, 0.5),
                )

                webp_buffer = io.BytesIO()
                image.save(
                    webp_buffer,
                    format="WEBP",
                    quality=WEBP_QUALITY,
                    optimize=True,
                    method=6,
                )
                webp_bytes = webp_buffer.getvalue()
            public_url = self.public_url
            if not public_url:
                raise RuntimeError("R2_PUBLIC_URL is not configured.")

            self.client.put_object(
                Bucket=self.bucket_name,
                Key=filename,
                Body=webp_bytes,
                ContentType="image/webp",
                CacheControl=CACHE_CONTROL,
            )
            return f"{public_url}/{filename}"
        except Exception as exc:
            print(
                f"[R2] cover upload failed for {slug!r}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            raise


r2_storage = R2ImageStorage()
