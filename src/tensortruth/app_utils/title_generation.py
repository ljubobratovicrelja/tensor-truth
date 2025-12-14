"""Title generation utilities using small LLM."""

import requests

from .logging_config import logger


def ensure_title_model_available():
    """Ensures the title generation model is available, pulling it if necessary."""
    title_model = "qwen2.5:0.5b"

    try:
        # Check if model exists
        resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            if title_model in models:
                return True

        # Model not found, pull it
        logger.info(f"Model {title_model} not found, pulling...")
        pull_payload = {"name": title_model, "stream": False}
        pull_resp = requests.post(
            "http://localhost:11434/api/pull", json=pull_payload, timeout=120
        )

        if pull_resp.status_code == 200:
            logger.info(f"Successfully pulled {title_model}")
            return True
        else:
            logger.error(f"Failed to pull model: {pull_resp.status_code}")
            return False
    except Exception as e:
        logger.error(f"Error checking/pulling model: {e}")
        return False


def generate_smart_title(text, model_name):
    """
    Uses a small, dedicated LLM to generate a concise title.
    Loads a tiny model (qwen2.5:0.5b), generates title, then unloads it.
    Returns the generated title or a truncated fallback.
    """
    # Use a tiny, fast model for title generation
    title_model = "qwen2.5:0.5b"

    # Ensure model is available (pull if needed)
    if not ensure_title_model_available():
        logger.warning("Title generation model unavailable, using fallback")
        return (text[:30] + "..") if len(text) > 30 else text

    try:
        # Prompt designed to minimize fluff
        prompt = f"Summarize this query into a concise 3-5 word title. Return ONLY the title text, no quotes. Query: {text}"

        payload = {
            "model": title_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": 512,  # Minimal context
                "num_predict": 15,  # Short answer
                "temperature": 0.3,
            },
            "keep_alive": 0,  # Unload immediately after generation
        }

        # Direct API call to avoid spinning up full engine logic
        resp = requests.post(
            "http://localhost:11434/api/generate", json=payload, timeout=10
        )
        if resp.status_code == 200:
            response = resp.json().get("response", "")

            # Final cleanup
            title = response.replace('"', "").replace("'", "").replace(".", "").strip()
            if title:
                logger.debug(f"Title generation success: '{title}'")
                return title
            else:
                logger.warning(f"Empty response after cleanup. Raw: {response[:100]}")
        else:
            logger.error(f"Title generation API returned status {resp.status_code}")
    except requests.exceptions.Timeout:
        logger.warning("Title generation timeout after 10s")
    except requests.exceptions.ConnectionError:
        logger.error("Connection error - is Ollama running?")
    except Exception as e:
        logger.error(f"Title generation error: {type(e).__name__}: {str(e)}")

    # Fallback
    logger.info(f"Using fallback title: '{text[:30]}...'")
    return (text[:30] + "..") if len(text) > 30 else text
