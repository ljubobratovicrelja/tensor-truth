"""Logging configuration for the application."""

import logging
import os
import sys

# Get log level from environment variable (default: INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
log_level_int = getattr(logging, log_level, logging.INFO)

# Configure logging
logging.basicConfig(
    level=log_level_int,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)
