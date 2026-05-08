"""
app/utils/logger.py
Configures structured logging for the whole application.
"""

import logging
import sys

from app.utils.config import get_settings

settings = get_settings()


def setup_logging() -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Quiet down noisy third-party libraries
    for noisy in ("httpx", "httpcore", "openai", "boto3", "botocore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
