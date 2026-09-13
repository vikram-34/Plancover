"""Gemini-backed JSON client for GMC policy extraction."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types


logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-3-flash"
DEFAULT_GEMINI_TIMEOUT_MS = 75_000


class JSONResponseError(ValueError):
    """Raised after all local-model attempts fail to produce a JSON object."""


class GeminiLLMClient:
    """Use Gemini's native structured output API for JSON model responses."""

    def __init__(
        self,
        *,
        model: str | None = None,
        timeout_ms: int = DEFAULT_GEMINI_TIMEOUT_MS,
        max_json_attempts: int = 3,
        client: genai.Client | None = None,
    ) -> None:
        if max_json_attempts < 2:
            raise ValueError("max_json_attempts must be at least 2")
        if timeout_ms < 1:
            raise ValueError("timeout_ms must be at least 1")
        load_dotenv()
        self.model = model or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        self.timeout_ms = timeout_ms
        self.max_json_attempts = max_json_attempts
        self.last_raw_response: str | None = None
        api_key = os.getenv("GEMINI_API_KEY")
        if client is None and not api_key:
            raise JSONResponseError(
                "GEMINI_API_KEY is missing. Add your Gemini API key to .env and retry."
            )
        self.client = client or genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=timeout_ms),
        )

    def complete_json(
        self,
        prompt: str,
        *,
        instructions: str,
        schema: dict[str, Any] | None = None,
        max_attempts: int | None = None,
    ) -> dict[str, Any]:
        """Request structured JSON and retry Gemini 429 responses."""

        attempt_limit = max_attempts or self.max_json_attempts
        if attempt_limit < 1:
            raise ValueError("max_attempts must be at least 1")
        last_error: JSONResponseError | None = None
        for attempt in range(1, attempt_limit + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=instructions,
                        response_mime_type="application/json",
                        response_schema=_gemini_response_schema(schema or {"type": "object"}),
                    ),
                )
            except errors.ClientError as exc:
                if getattr(exc, "code", None) == 429 and attempt < attempt_limit:
                    delay = 2 ** (attempt - 1)
                    logger.warning(
                        "Gemini rate limit reached (attempt %s/%s); retrying in %ss.",
                        attempt,
                        attempt_limit,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                raise JSONResponseError(f"Gemini API request failed: {exc}") from exc
            content = response.text
            if not content:
                raise JSONResponseError("Gemini returned an empty response")
            self.last_raw_response = content
            try:
                return _parse_json_object(content)
            except JSONResponseError as exc:
                last_error = exc
                if attempt < attempt_limit:
                    logger.warning(
                        "Gemini response was not valid JSON (attempt %s/%s); retrying.",
                        attempt,
                        attempt_limit,
                    )

        raise JSONResponseError(
            f"Gemini did not return valid JSON after {attempt_limit} attempts: {last_error}"
        )


class JSONDecodeError(JSONResponseError):
    """Internal parse error type retained for the retry loop."""


def _gemini_response_schema(
    schema: dict[str, Any], definitions: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Remove JSON Schema keywords unsupported by Gemini structured output."""

    if isinstance(schema, list):
        return [_gemini_response_schema(item, definitions) for item in schema]
    if not isinstance(schema, dict):
        return schema
    definitions = definitions or schema.get("$defs", {})

    reference = schema.get("$ref")
    if reference and reference.startswith("#/$defs/"):
        definition_name = reference.removeprefix("#/$defs/")
        return _gemini_response_schema(definitions[definition_name], definitions)

    any_of = schema.get("anyOf")
    if isinstance(any_of, list):
        non_null = [item for item in any_of if item.get("type") != "null"]
        if len(non_null) == 1 and len(non_null) != len(any_of):
            converted = _gemini_response_schema(non_null[0], definitions)
            converted["nullable"] = True
            return converted

    converted: dict[str, Any] = {}
    for key, value in schema.items():
        if key in {"additionalProperties", "$defs", "$ref", "title", "default", "anyOf"}:
            continue
        if key == "type" and isinstance(value, list):
            nullable = "null" in value
            non_null_types = [item for item in value if item != "null"]
            if non_null_types:
                converted["type"] = non_null_types[0]
                if nullable:
                    converted["nullable"] = True
            continue
        converted[key] = _gemini_response_schema(value, definitions)
    return converted


def _parse_json_object(content: str) -> dict[str, Any]:
    """Parse raw JSON, then retry once after removing a Markdown code fence."""

    for candidate in (content.strip(), _strip_markdown_fence(content)):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            raise JSONDecodeError("Gemini response must be a JSON object")
        return parsed
    raise JSONDecodeError("Gemini response is not valid JSON")


def _strip_markdown_fence(content: str) -> str:
    """Return the content inside one optional ```json or ``` fence."""

    stripped = content.strip()
    match = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", stripped, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else stripped
