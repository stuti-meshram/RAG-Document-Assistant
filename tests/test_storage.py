"""
tests/test_storage.py
Unit tests for S3Handler using moto (AWS mock library).
"""

import pytest
import boto3
from moto import mock_aws

from app.utils.config import get_settings

settings = get_settings()


@pytest.fixture(autouse=True)
def aws_credentials(monkeypatch):
    """Set fake AWS credentials so moto works without real creds."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SECURITY_TOKEN", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


@mock_aws
class TestS3Handler:
    """
    All tests run inside @mock_aws so no real S3 calls are made.
    We import S3Handler inside each test to ensure the boto3 client
    is created after moto patches the AWS SDK.
    """

    def _make_handler(self):
        # Patch the bucket name to avoid conflicts
        import os
        os.environ["S3_BUCKET_NAME"] = "test-rag-bucket"
        # Reset the settings cache so new env var is picked up
        from app.utils.config import get_settings
        get_settings.cache_clear()
        from app.storage.s3_handler import S3Handler
        return S3Handler()

    def test_upload_and_download(self):
        handler = self._make_handler()
        content = b"Hello, this is a PDF content mock."
        key = handler.upload(content, "test.pdf")
        assert "test.pdf" in key

        downloaded = handler.download(key)
        assert downloaded == content

    def test_list_documents(self):
        handler = self._make_handler()
        handler.upload(b"doc1", "a.pdf")
        handler.upload(b"doc2", "b.pdf")

        docs = handler.list_documents()
        filenames = [d["filename"] for d in docs]
        assert "a.pdf" in filenames
        assert "b.pdf" in filenames

    def test_delete(self):
        handler = self._make_handler()
        key = handler.upload(b"temporary", "temp.pdf")
        assert handler.delete(key)

        docs = handler.list_documents()
        assert not any(d["key"] == key for d in docs)

    def test_is_available(self):
        handler = self._make_handler()
        assert handler.is_available()

    def test_presigned_url(self):
        handler = self._make_handler()
        key = handler.upload(b"data", "report.pdf")
        url = handler.get_presigned_url(key, expiry_seconds=60)
        assert url is not None
        assert "report.pdf" in url or "X-Amz" in url
