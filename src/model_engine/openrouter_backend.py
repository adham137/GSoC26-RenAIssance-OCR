"""
src/model_engine/openrouter_backend.py
=======================================
OpenRouter API Backend Implementation.

This module provides a synchronous backend that calls OpenRouter's API
for vision-language inference. It implements the BaseModelBackend interface
so it can be used interchangeably with the local ModelExecutor.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from pathlib import Path
from typing import Any

import requests

from src.model_engine.base_backend import BaseModelBackend
from src.models import DocumentPage

logger = logging.getLogger(__name__)


class OpenRouterAPIError(Exception):
    """Raised when the OpenRouter API returns an error response."""

    pass


class OpenRouterAuthError(OpenRouterAPIError):
    """Raised specifically on HTTP 401 — bad or missing API key."""

    pass


class OpenRouterBackend(BaseModelBackend):
    """
    Backend that uses OpenRouter's API for inference.

    Parameters
    ----------
    config : dict
        Configuration dictionary with the following keys:
        - api_key (str): OpenRouter API key (required)
        - model (str): OpenRouter model identifier (required)
        - max_tokens (int): Maximum tokens for completion (default: 2048)
        - temperature (float): Sampling temperature (default: 0.1)
        - timeout_seconds (int): HTTP request timeout (default: 60)
        - max_retries (int): Number of retry attempts (default: 3)
        - retry_delay_seconds (float): Base delay between retries (default: 2.0)
        - site_url (str, optional): HTTP-Referer header (default: GitHub URL)
        - app_name (str, optional): X-Title header (default: framework name)

    Raises
    ------
    EnvironmentError
        If OPENROUTER_API_KEY is not set in environment.
    """

    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    # Supported image formats
    SUPPORTED_IMAGE_TYPES = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }

    def __init__(self, config: dict) -> None:
        # Validate required config
        if "api_key" not in config:
            raise KeyError("Missing required config key: 'api_key'")
        if "model" not in config:
            raise KeyError("Missing required config key: 'model'")

        # Resolve API key from environment if using ${...} syntax
        api_key = config["api_key"]
        if api_key.startswith("${") and api_key.endswith("}"):
            env_var_name = api_key[2:-1]
            api_key = os.environ.get(env_var_name)
            if api_key is None:
                raise EnvironmentError(
                    f"{env_var_name} environment variable is not set."
                )

        self._api_key = api_key
        self._model = config.get("model", "qwen/qwen2.5-vl-72b-instruct:free")
        self._max_tokens = config.get("max_tokens", 2048)
        self._temperature = config.get("temperature", 0.1)
        self._timeout_seconds = config.get("timeout_seconds", 60)
        self._max_retries = config.get("max_retries", 3)
        self._retry_delay_seconds = config.get("retry_delay_seconds", 2.0)
        self._site_url = config.get(
            "site_url", "https://github.com/humanai-renaissance"
        )
        self._app_name = config.get("app_name", "Agentic-OCR-Framework")

        # Log initialization
        logger.info(
            "OpenRouter backend initialized — model=%s, adapter=unavailable (API backend)",
            self._model,
        )

    def _encode_image(self, image_path: str) -> str:
        """
        Encode an image file as base64, resizing if necessary to prevent
        excessive token usage.

        Parameters
        ----------
        image_path : str
            Path to the image file.

        Returns
        -------
        str
            Base64-encoded image string.

        Raises
        ------
        FileNotFoundError
            If the image file does not exist.
        """
        from PIL import Image
        from io import BytesIO

        from src.model_engine.model_executor import ModelExecutor

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Open and resize image if it exceeds MAX_IMAGE_DIM
        with Image.open(path) as img:
            img = img.convert("RGB")  # Ensure RGB format
            width, height = img.size
            max_dim = max(width, height)

            if max_dim > ModelExecutor.MAX_IMAGE_DIM:
                scale = ModelExecutor.MAX_IMAGE_DIM / max_dim
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.info(
                    "Resized image from %dx%d to %dx%d (max_dim=%d)",
                    width,
                    height,
                    new_width,
                    new_height,
                    ModelExecutor.MAX_IMAGE_DIM,
                )

            # Encode to JPEG for smaller file size
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            image_bytes = buffer.getvalue()

        return base64.b64encode(image_bytes).decode("utf-8")

    def _get_media_type(self, image_path: str) -> str:
        """
        Get the MIME media type for an image file.

        Parameters
        ----------
        image_path : str
            Path to the image file.

        Returns
        -------
        str
            MIME type (e.g., "image/png").

        Raises
        ------
        ValueError
            If the file extension is not supported.
        """
        path = Path(image_path)
        ext = path.suffix.lower()

        if ext not in self.SUPPORTED_IMAGE_TYPES:
            raise ValueError(
                f"Unsupported image format: {ext}. "
                f"Supported formats: {', '.join(self.SUPPORTED_IMAGE_TYPES.keys())}. "
                f"Image path: {image_path}"
            )

        return self.SUPPORTED_IMAGE_TYPES[ext]

    def _call_api(self, messages: list[dict]) -> str:
        """
        Make an API call to OpenRouter with retry logic.

        Parameters
        ----------
        messages : list[dict]
            OpenRouter-compatible messages list.

        Returns
        -------
        str
            Model response content.

        Raises
        ------
        OpenRouterAuthError
            On HTTP 401 (invalid API key).
        OpenRouterAPIError
            On other HTTP errors or after all retries exhausted.
        """
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self._site_url,
            "X-Title": self._app_name,
        }

        payload = {
            "model": self._model,
            "messages": messages,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }

        # Get prompt length for logging (from first message content)
        prompt_chars = 0
        if messages and "content" in messages[0]:
            content = messages[0]["content"]
            if isinstance(content, str):
                prompt_chars = len(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        prompt_chars += len(item.get("text", ""))

        last_exception: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                # Log request details
                logger.info(
                    "📤 Sending request to OpenRouter API (attempt %d/%d) — model=%s, max_tokens=%d",
                    attempt + 1,
                    self._max_retries,
                    self._model,
                    self._max_tokens,
                )

                response = requests.post(
                    self.API_URL,
                    headers=headers,
                    json=payload,
                    timeout=self._timeout_seconds,
                )

                if response.status_code == 401:
                    # Mask API key in logs - show first 8 chars only
                    key_preview = (
                        self._api_key[:8] + "****" if len(self._api_key) > 8 else "****"
                    )
                    raise OpenRouterAuthError(
                        f"HTTP 401: Invalid API key ({key_preview})"
                    )

                if response.status_code == 400:
                    raise OpenRouterAPIError(
                        f"HTTP 400: Bad request. Response: {response.text}"
                    )

                if response.status_code in (429, 500, 502, 503, 504):
                    # Transient error - retry with exponential backoff
                    if attempt < self._max_retries - 1:
                        delay = self._retry_delay_seconds * (2**attempt)
                        logger.warning(
                            "OpenRouter retry %d/%d after %s",
                            attempt + 1,
                            self._max_retries,
                            response.status_code,
                        )
                        time.sleep(delay)
                        continue
                    else:
                        raise OpenRouterAPIError(
                            f"HTTP {response.status_code}: API error after {self._max_retries} retries. "
                            f"Response: {response.text}"
                        )

                response.raise_for_status()

                # Extract response content with null guard
                result = response.json()
                content = result["choices"][0]["message"].get("content")

                if content is None:
                    finish_reason = result["choices"][0].get("finish_reason", "unknown")
                    raise OpenRouterAPIError(
                        f"Model returned null content (finish_reason='{finish_reason}'). "
                        "The model may have triggered a content filter, returned a tool call "
                        f"instead of text, or the response structure was unexpected. "
                        f"Full choice: {result['choices'][0]}"
                    )

                # Extract usage info if available
                usage_info = result.get("usage", {})
                prompt_tokens = usage_info.get("prompt_tokens", "?")
                completion_tokens = usage_info.get("completion_tokens", "?")

                # Log successful response details
                logger.info(
                    "📥 Received response from OpenRouter — status=%d, prompt_tokens=%s, completion_tokens=%s, total_chars=%d",
                    response.status_code,
                    prompt_tokens,
                    completion_tokens,
                    len(content) if content is not None else 0,
                )

                return content

            except requests.exceptions.Timeout as e:
                last_exception = e
                if attempt < self._max_retries - 1:
                    delay = self._retry_delay_seconds * (2**attempt)
                    logger.warning(
                        "OpenRouter retry %d/%d after timeout",
                        attempt + 1,
                        self._max_retries,
                    )
                    time.sleep(delay)
                else:
                    raise OpenRouterAPIError(
                        f"Request timed out after {self._max_retries} retries"
                    ) from e

            except requests.exceptions.ConnectionError as e:
                last_exception = e
                if attempt < self._max_retries - 1:
                    delay = self._retry_delay_seconds * (2**attempt)
                    logger.warning(
                        "OpenRouter retry %d/%d after connection error",
                        attempt + 1,
                        self._max_retries,
                    )
                    time.sleep(delay)
                else:
                    raise OpenRouterAPIError(
                        f"Connection error after {self._max_retries} retries"
                    ) from e

            except OpenRouterAuthError:
                # Don't retry auth errors
                raise

            except OpenRouterAPIError:
                # Don't retry bad request errors
                raise

            except Exception as e:
                last_exception = e
                if attempt < self._max_retries - 1:
                    delay = self._retry_delay_seconds * (2**attempt)
                    logger.warning(
                        "OpenRouter retry %d/%d after %s",
                        attempt + 1,
                        self._max_retries,
                        type(e).__name__,
                    )
                    time.sleep(delay)
                else:
                    raise OpenRouterAPIError(
                        f"Unexpected error after {self._max_retries} retries: {e}"
                    ) from e

        # Should not reach here, but just in case
        raise OpenRouterAPIError(f"API call failed after all retries: {last_exception}")

    def _build_vision_message(self, prompt: str, image_path: str) -> list[dict]:
        """
        Build an OpenRouter-compatible vision message.

        Parameters
        ----------
        prompt : str
            Text prompt.
        image_path : str
            Path to the image file.

        Returns
        -------
        list[dict]
            Messages list for the API.
        """
        base64_image = self._encode_image(image_path)
        media_type = self._get_media_type(image_path)

        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{base64_image}"
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

    def _build_text_message(self, prompt: str) -> list[dict]:
        """
        Build an OpenRouter-compatible text-only message.

        Parameters
        ----------
        prompt : str
            Text prompt.

        Returns
        -------
        list[dict]
            Messages list for the API.
        """
        return [
            {
                "role": "user",
                "content": prompt,
            }
        ]

    def extract_text(self, page: DocumentPage, prompt: str) -> str:
        """
        Extract text from a document page image.

        Parameters
        ----------
        page : DocumentPage
            The document page to process.
        prompt : str
            The OCR prompt.

        Returns
        -------
        str
            Extracted transcription text.
        """
        messages = self._build_vision_message(prompt, page.image_path)
        return self._call_api(messages)

    def diagnose_errors(
        self, page: DocumentPage, current_text: str, prompt: str
    ) -> str:
        """
        Identify transcription errors (text-only reasoning task).

        Parameters
        ----------
        page : DocumentPage
            The document page (not used - text-only mode).
        current_text : str
            Current transcription text.
        prompt : str
            The reflection prompt.

        Returns
        -------
        str
            Error diagnosis and correction plan.
        """
        messages = self._build_text_message(prompt)
        return self._call_api(messages)

    def filter_plan(self, raw_plan: str, prompt: str) -> str:
        """
        Filter infeasible corrections from a raw plan (text-only).

        Parameters
        ----------
        raw_plan : str
            Raw correction plan.
        prompt : str
            The filtering prompt.

        Returns
        -------
        str
            Feasible subset of the correction plan.
        """
        messages = self._build_text_message(prompt)
        return self._call_api(messages)

    def guided_refinement(
        self, page: DocumentPage, feasible_plan: str, prompt: str
    ) -> str:
        """
        Refine transcription using a feasible plan.

        Parameters
        ----------
        page : DocumentPage
            The document page to process.
        feasible_plan : str
            Filtered correction plan.
        prompt : str
            The refinement prompt.

        Returns
        -------
        str
            Refined transcription text.
        """
        messages = self._build_vision_message(prompt, page.image_path)
        return self._call_api(messages)

    def set_adapter(self) -> None:
        """No-op for API backend."""
        pass

    def disable_adapter(self) -> None:
        """No-op for API backend."""
        pass

    def is_adapter_available(self) -> bool:
        """
        API models never have adapters.

        Returns
        -------
        bool
            Always False.
        """
        return False

    def _parse_transcription(self, raw_response: str) -> str:
        """
        Extract the clean transcription from a raw model response.

        Handles three common response patterns:
        1. <think>...</think> tags followed by the transcription
        2. Markdown separator (***  or ---) followed by the transcription
        3. Introductory prose paragraph followed by the transcription
           (detected by a blank line or line starting with an uppercase word)

        Parameters
        ----------
        raw_response : str
            The full raw output from the model.

        Returns
        -------
        str
            The extracted transcription text, stripped of all preamble.
        """
        import re

        text = raw_response.strip()

        # Pattern 1 — explicit </think> tag (Qwen3 thinking models)
        think_match = re.search(r"</think>\s*(.*)", text, re.DOTALL | re.IGNORECASE)
        if think_match:
            return think_match.group(1).strip()

        # Pattern 2 — markdown horizontal rule separator (*** or --- or ___)
        # Everything after the LAST separator is the transcription
        separator_match = re.split(r"\n\s*(\*{3,}|_{3,}|-{3,})\s*\n", text)
        if len(separator_match) > 1:
            return separator_match[-1].strip()

        # Pattern 3 — skip lines that look like introductory prose
        # Heuristic: prose lines tend to be long sentences ending in punctuation.
        # Transcription lines tend to be short fragments or ALL CAPS.
        lines = text.splitlines()
        transcription_start = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            # A line looks like prose if it's a complete sentence (ends in . or ,)
            # and is longer than 60 chars — skip it
            is_prose = len(stripped) > 60 and stripped[-1] in ".,"
            if not is_prose:
                transcription_start = i
                break

        return "\n".join(lines[transcription_start:]).strip()

    @property
    def adapter_state(self) -> str:
        """
        API models never have adapters.

        Returns
        -------
        str
            Always "OFF".
        """
        return "OFF"
