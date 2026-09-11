"""Ollama-backed JSON client for local GMC policy extraction."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_OLLAMA_MODEL = "llama3"


class JSONResponseError(ValueError):
    """Raised after all local-model attempts fail to produce a JSON object."""


class OllamaLLMClient:
    """Use Ollama's OpenAI-compatible API for JSON-only model responses."""

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        max_json_attempts: int = 3,
        client: OpenAI | None = None,
    ) -> None:
        if max_json_attempts < 2:
            raise ValueError("max_json_attempts must be at least 2")
        load_dotenv()
        self.model = model or os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        self.max_json_attempts = max_json_attempts
        self.last_raw_response: str | None = None
        # Ollama's local endpoint does not authenticate, but the OpenAI SDK
        # requires a non-empty API key for its compatible client.
        self.client = client or OpenAI(base_url=base_url, api_key="ollama")

    def complete_json(
        self,
        prompt: str,
        *,
        instructions: str,
        schema: dict[str, Any] | None = None,
        max_attempts: int | None = None,
    ) -> dict[str, Any]:
        """Request JSON, repairing fenced JSON once on every parse failure.

        Local models sometimes wrap otherwise-valid JSON in Markdown despite the
        instruction.  Each model response is parsed first as-is, then once more
        after stripping a surrounding code fence.  A new completion is requested
        only when both parses fail, up to ``max_json_attempts`` total attempts.
        """

        attempt_limit = max_attempts or self.max_json_attempts
        if attempt_limit < 1:
            raise ValueError("max_attempts must be at least 1")
        last_error: JSONResponseError | None = None
        for attempt in range(1, attempt_limit + 1):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": prompt},
                ],
                # Ollama supports the OpenAI JSON-object response format for
                # compatible local models. The prompt also carries the schema.
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            if not content:
                raise JSONResponseError("Ollama returned an empty response")
            self.last_raw_response = content

            try:
                return _parse_json_object(content)
            except JSONResponseError as exc:
                last_error = exc
                if attempt < self.max_json_attempts:
                    logger.warning(
                        "Ollama response was not valid JSON (attempt %s/%s); retrying.",
                        attempt,
                        attempt_limit,
                    )

        raise JSONResponseError(
            f"Ollama did not return valid JSON after {attempt_limit} attempts: {last_error}"
        )


class JSONDecodeError(JSONResponseError):
    """Internal parse error type retained for the retry loop."""


def _parse_json_object(content: str) -> dict[str, Any]:
    """Parse raw JSON, then retry once after removing a Markdown code fence."""

    for candidate in (content.strip(), _strip_markdown_fence(content)):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            raise JSONDecodeError("Ollama response must be a JSON object")
        return parsed
    raise JSONDecodeError("Ollama response is not valid JSON")


def _strip_markdown_fence(content: str) -> str:
    """Return the content inside one optional ```json or ``` fence."""

    stripped = content.strip()
    match = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", stripped, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else stripped
