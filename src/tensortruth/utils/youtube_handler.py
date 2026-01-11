"""YouTube-specific content handler for web search.

Uses youtube-transcript-api to fetch video metadata and transcripts.
"""

import asyncio
import logging
import re
from typing import Optional, Tuple
from urllib.parse import parse_qs, urlparse

import aiohttp
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

from .domain_handlers import ContentHandler, register_handler

logger = logging.getLogger(__name__)


class YouTubeHandler(ContentHandler):
    """Handler for YouTube video URLs."""

    @property
    def name(self) -> str:
        return "YouTube"

    def matches(self, url: str) -> bool:
        """Check if URL is a YouTube video."""
        try:
            parsed = urlparse(url)
            # Match youtube.com or youtu.be domains
            if "youtube.com" in parsed.netloc or "youtu.be" in parsed.netloc:
                # Must have a video ID (v parameter or path for youtu.be)
                video_id = self._extract_video_id(url)
                return video_id is not None
            return False
        except Exception:
            return False

    def _extract_video_id(self, url: str) -> Optional[str]:
        """
        Extract YouTube video ID from URL.

        Supports formats:
        - https://www.youtube.com/watch?v=VIDEO_ID
        - https://youtu.be/VIDEO_ID
        - https://www.youtube.com/embed/VIDEO_ID
        - https://m.youtube.com/watch?v=VIDEO_ID

        Args:
            url: YouTube URL

        Returns:
            Video ID or None if extraction fails
        """
        try:
            parsed = urlparse(url)

            # youtu.be format
            if "youtu.be" in parsed.netloc:
                # Path is /VIDEO_ID
                video_id = parsed.path.strip("/").split("?")[0]
                if video_id:
                    return video_id

            # youtube.com formats
            if "youtube.com" in parsed.netloc:
                # /watch?v=VIDEO_ID
                if parsed.path == "/watch":
                    query_params = parse_qs(parsed.query)
                    if "v" in query_params:
                        return query_params["v"][0]

                # /embed/VIDEO_ID
                if parsed.path.startswith("/embed/"):
                    video_id = parsed.path.split("/embed/")[1].split("?")[0]
                    if video_id:
                        return video_id

                # /v/VIDEO_ID (older format)
                if parsed.path.startswith("/v/"):
                    video_id = parsed.path.split("/v/")[1].split("?")[0]
                    if video_id:
                        return video_id

        except Exception as e:
            logger.warning(f"Failed to extract video ID from {url}: {e}")

        return None

    def _extract_metadata_from_html(self, html: str) -> dict:
        """
        Extract video metadata from YouTube HTML page.

        Args:
            html: YouTube page HTML

        Returns:
            Dict with title, channel, upload_date, description
        """
        metadata = {
            "title": None,
            "channel": None,
            "upload_date": None,
            "description": None,
        }

        try:
            # Extract title from og:title meta tag
            title_match = re.search(
                r'<meta property="og:title" content="([^"]+)"', html
            )
            if title_match:
                metadata["title"] = title_match.group(1)

            # Extract channel from channelName
            channel_match = re.search(r'"channelName":"([^"]+)"', html) or re.search(
                r'"author":"([^"]+)"', html
            )
            if channel_match:
                metadata["channel"] = channel_match.group(1)

            # Extract upload date
            date_match = re.search(r'"uploadDate":"([^"]+)"', html)
            if date_match:
                metadata["upload_date"] = date_match.group(1).split("T")[0]

            # Extract description
            desc_match = re.search(
                r'<meta property="og:description" content="([^"]+)"', html
            )
            if desc_match:
                metadata["description"] = desc_match.group(1)

        except Exception as e:
            logger.warning(f"Failed to extract some metadata: {e}")

        return metadata

    async def fetch(
        self, url: str, session: aiohttp.ClientSession, timeout: int = 10
    ) -> Tuple[Optional[str], str, Optional[str]]:
        """
        Fetch YouTube video metadata and transcript.

        Args:
            url: YouTube video URL
            session: aiohttp ClientSession
            timeout: Timeout in seconds

        Returns:
            Tuple of (markdown_content, status, error_message)
        """
        logger.info(f"Fetching YouTube video: {url}")

        # Extract video ID
        video_id = self._extract_video_id(url)
        if not video_id:
            return None, "parse_error", "Could not extract video ID from URL"

        try:
            # Fetch video page for metadata
            timeout_obj = aiohttp.ClientTimeout(total=timeout)
            async with session.get(url, timeout=timeout_obj) as response:
                if response.status != 200:
                    return (
                        None,
                        "http_error",
                        f"HTTP {response.status}",
                    )
                html = await response.text()

            # Extract metadata from HTML
            metadata = self._extract_metadata_from_html(html)

            # Fetch transcript in executor (blocking API call)
            loop = asyncio.get_event_loop()

            def get_transcript():
                try:
                    # Try to get transcript (prefer manually created ones)
                    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

                    # Try manual transcripts first
                    try:
                        transcript = transcript_list.find_manually_created_transcript(
                            ["en"]
                        )
                    except NoTranscriptFound:
                        # Fall back to auto-generated
                        transcript = transcript_list.find_generated_transcript(["en"])

                    # Get the actual transcript data
                    return transcript.fetch()
                except (
                    NoTranscriptFound,
                    TranscriptsDisabled,
                    VideoUnavailable,
                ) as e:
                    logger.warning(f"Transcript not available for {video_id}: {e}")
                    return None

            transcript_data = await asyncio.wait_for(
                loop.run_in_executor(None, get_transcript), timeout=timeout
            )

            # Build markdown content
            markdown_lines = []

            # Add title
            title = metadata.get("title") or f"YouTube Video {video_id}"
            markdown_lines.append(f"# {title}")
            markdown_lines.append("")

            # Add source metadata
            markdown_lines.append(f"<!-- Source: {url} -->")
            markdown_lines.append(f"<!-- Video ID: {video_id} -->")
            markdown_lines.append("")

            # Add metadata section
            markdown_lines.append("## Video Information")
            markdown_lines.append("")
            if metadata.get("channel"):
                markdown_lines.append(f"**Channel**: {metadata['channel']}")
            if metadata.get("upload_date"):
                markdown_lines.append(f"**Upload Date**: {metadata['upload_date']}")
            markdown_lines.append(f"**Video URL**: {url}")
            markdown_lines.append("")

            # Add description if available
            if metadata.get("description"):
                markdown_lines.append("## Description")
                markdown_lines.append("")
                markdown_lines.append(metadata["description"])
                markdown_lines.append("")

            # Add transcript if available
            if transcript_data:
                markdown_lines.append("## Transcript")
                markdown_lines.append("")

                # Combine transcript segments
                # Limit transcript size to prevent overwhelming content
                MAX_TRANSCRIPT_CHARS = 8000
                transcript_text = " ".join(
                    [segment["text"] for segment in transcript_data]
                )

                if len(transcript_text) > MAX_TRANSCRIPT_CHARS:
                    transcript_text = (
                        transcript_text[:MAX_TRANSCRIPT_CHARS]
                        + "\n\n[Transcript truncated for length...]"
                    )

                markdown_lines.append(transcript_text)
                markdown_lines.append("")
            else:
                markdown_lines.append(
                    "*Note: Transcript not available for this video.*"
                )
                markdown_lines.append("")

            markdown = "\n".join(markdown_lines)

            # Quality check
            if len(markdown.strip()) < 100:
                return None, "too_short", "YouTube content too short"

            logger.info(f"✅ Fetched YouTube video '{title}' ({len(markdown)} chars)")
            return markdown, "success", None

        except asyncio.TimeoutError:
            error_msg = "YouTube fetch timeout"
            logger.warning(f"Timeout fetching YouTube video: {video_id}")
            return None, "timeout", error_msg

        except Exception as e:
            error_msg = f"YouTube fetch error: {str(e)}"
            logger.error(f"Error fetching YouTube video {video_id}: {e}")
            return None, "parse_error", error_msg


# Register the YouTube handler
register_handler(YouTubeHandler())
