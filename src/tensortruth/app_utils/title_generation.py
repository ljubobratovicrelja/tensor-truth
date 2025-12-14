"""Title generation utilities using small LLM."""

import asyncio

import aiohttp

from .logging_config import logger


async def ensure_title_model_available_async():
    """Ensures the title generation model is available, pulling it if necessary (async version)."""
    title_model = "qwen2.5:0.5b"

    try:
        timeout = aiohttp.ClientTimeout(total=2)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Check if model exists
            async with session.get("http://localhost:11434/api/tags") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models = [m["name"] for m in data.get("models", [])]
                    if title_model in models:
                        return True

            # Model not found, pull it
            logger.info(f"Model {title_model} not found, pulling...")
            pull_payload = {"name": title_model, "stream": False}

            pull_timeout = aiohttp.ClientTimeout(total=120)
            async with aiohttp.ClientSession(timeout=pull_timeout) as pull_session:
                async with pull_session.post(
                    "http://localhost:11434/api/pull", json=pull_payload
                ) as pull_resp:
                    if pull_resp.status == 200:
                        logger.info(f"Successfully pulled {title_model}")
                        return True
                    else:
                        logger.error(f"Failed to pull model: {pull_resp.status}")
                        return False
    except Exception as e:
        logger.error(f"Error checking/pulling model: {e}")
        return False


def ensure_title_model_available():
    """Ensures the title generation model is available, pulling it if necessary (sync wrapper)."""
    return asyncio.run(ensure_title_model_available_async())


async def generate_smart_title_async(text, model_name="qwen2.5:0.5b"):
    """
    Uses a small, dedicated LLM to generate a concise title (async version).
    Loads a tiny model (qwen2.5:0.5b), generates title, then unloads it.
    Returns the generated title or a truncated fallback.

    Args:
        text: The text to generate a title for
        model_name: Optional model name (currently unused, kept for API compatibility)
    """
    # Ensure model is available (pull if needed)
    if not await ensure_title_model_available_async():
        logger.warning("Title generation model unavailable, using fallback")
        return (text[:30] + "..") if len(text) > 30 else text

    try:
        # Prompt designed to minimize fluff
        prompt = (
            f"Summarize this query into a concise 3-5 word title. "
            f"Return ONLY the title text, no quotes. Query: {text}"
        )

        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": 512,  # Minimal context
                "num_predict": 15,  # Short answer
                "temperature": 0.3,
            },
            "keep_alive": 0,  # Unload immediately after generation
        }

        # Direct API call to avoid spinning up full engine logic (async)
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                "http://localhost:11434/api/generate", json=payload
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    response = data.get("response", "")

                    # Final cleanup
                    title = (
                        response.replace('"', "")
                        .replace("'", "")
                        .replace(".", "")
                        .strip()
                    )
                    if title:
                        logger.debug(f"Title generation success: '{title}'")
                        return title
                    else:
                        logger.warning(
                            f"Empty response after cleanup. Raw: {response[:100]}"
                        )
                else:
                    logger.error(f"Title generation API returned status {resp.status}")
    except asyncio.TimeoutError:
        logger.warning("Title generation timeout after 10s")
    except aiohttp.ClientError as e:
        logger.error(f"Connection error - is Ollama running? {e}")
    except Exception as e:
        logger.error(f"Title generation error: {type(e).__name__}: {str(e)}")

    # Fallback
    logger.info(f"Using fallback title: '{text[:30]}...'")
    return (text[:30] + "..") if len(text) > 30 else text


def generate_smart_title(text, model_name="qwen2.5:0.5b"):
    """
    Uses a small, dedicated LLM to generate a concise title (sync wrapper).
    Loads a tiny model (qwen2.5:0.5b), generates title, then unloads it.
    Returns the generated title or a truncated fallback.

    Args:
        text: The text to generate a title for
        model_name: Optional model name (currently unused, kept for API compatibility)
    """
    return asyncio.run(generate_smart_title_async(text, model_name))
