"""
app/storage/s3_handler.py
Thin wrapper around boto3 for uploading / downloading PDF files to S3.
"""

import logging
from io import BytesIO
from typing import Optional

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from app.utils.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class S3Handler:
    """
    Manages PDF persistence in AWS S3.

    Usage
    -----
    handler = S3Handler()
    key = handler.upload(file_bytes, "report.pdf")
    data = handler.download(key)
    all_keys = handler.list_documents()
    """

    def __init__(self):
        try:
            self._client = boto3.client(
                "s3",
                region_name=settings.aws_region,
                aws_access_key_id=settings.aws_access_key_id or None,
                aws_secret_access_key=settings.aws_secret_access_key or None,
            )
            self._bucket = settings.s3_bucket_name
            self._prefix = settings.s3_prefix.rstrip("/") + "/"
            self._ensure_bucket()
        except NoCredentialsError:
            logger.warning(
                "AWS credentials not found. S3 features will be disabled."
            )
            self._client = None

    # ─── Public API ──────────────────────────────────────────────────────

    def upload(self, file_bytes: bytes, filename: str) -> str:
        """Upload *file_bytes* and return the S3 object key."""
        if not self.is_available():
            raise RuntimeError("S3 is not configured / available.")

        key = f"{self._prefix}{filename}"
        try:
            self._client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=file_bytes,
                ContentType="application/pdf",
                Metadata={"original-filename": filename},
            )
            logger.info("Uploaded '%s' → s3://%s/%s", filename, self._bucket, key)
            return key
        except ClientError as exc:
            logger.error("S3 upload failed for '%s': %s", filename, exc)
            raise

    def download(self, s3_key: str) -> bytes:
        """Download an object by its full key and return raw bytes."""
        if not self.is_available():
            raise RuntimeError("S3 is not configured / available.")

        try:
            response = self._client.get_object(Bucket=self._bucket, Key=s3_key)
            data: bytes = response["Body"].read()
            logger.info("Downloaded s3://%s/%s (%d bytes)", self._bucket, s3_key, len(data))
            return data
        except ClientError as exc:
            logger.error("S3 download failed for '%s': %s", s3_key, exc)
            raise

    def list_documents(self) -> list[dict]:
        """
        Return a list of uploaded documents.
        Each entry: {key, filename, size_bytes, last_modified}
        """
        if not self.is_available():
            return []

        try:
            paginator = self._client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self._bucket, Prefix=self._prefix)
            results: list[dict] = []
            for page in pages:
                for obj in page.get("Contents", []):
                    key: str = obj["Key"]
                    if key == self._prefix:  # skip the folder placeholder
                        continue
                    results.append(
                        {
                            "key": key,
                            "filename": key.replace(self._prefix, ""),
                            "size_bytes": obj["Size"],
                            "last_modified": obj["LastModified"].isoformat(),
                        }
                    )
            return results
        except ClientError as exc:
            logger.error("S3 list failed: %s", exc)
            return []

    def delete(self, s3_key: str) -> bool:
        """Delete an object from S3. Returns True on success."""
        if not self.is_available():
            return False

        try:
            self._client.delete_object(Bucket=self._bucket, Key=s3_key)
            logger.info("Deleted s3://%s/%s", self._bucket, s3_key)
            return True
        except ClientError as exc:
            logger.error("S3 delete failed for '%s': %s", s3_key, exc)
            return False

    def get_presigned_url(self, s3_key: str, expiry_seconds: int = 3600) -> Optional[str]:
        """Generate a pre-signed download URL (default 1-hour expiry)."""
        if not self.is_available():
            return None

        try:
            url: str = self._client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self._bucket, "Key": s3_key},
                ExpiresIn=expiry_seconds,
            )
            return url
        except ClientError as exc:
            logger.error("Presigned URL generation failed: %s", exc)
            return None

    def is_available(self) -> bool:
        return self._client is not None

    # ─── Private helpers ─────────────────────────────────────────────────

    def _ensure_bucket(self) -> None:
        """Create the bucket if it does not exist (dev convenience)."""
        try:
            self._client.head_bucket(Bucket=self._bucket)
        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            if code in ("404", "NoSuchBucket"):
                logger.info("Creating S3 bucket '%s'…", self._bucket)
                kwargs: dict = {"Bucket": self._bucket}
                if settings.aws_region != "us-east-1":
                    kwargs["CreateBucketConfiguration"] = {
                        "LocationConstraint": settings.aws_region
                    }
                self._client.create_bucket(**kwargs)
            else:
                raise
