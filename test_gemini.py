"""Minimal Gemini connectivity probe.

Run with: ``python test_gemini.py``
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


def main() -> None:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is missing from .env")

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=75_000),
    )
    response = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-3-flash"),
        contents="Return the JSON object {\"status\": \"ok\"}.",
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema={"type": "object", "properties": {"status": {"type": "string"}}},
        ),
    )
    print(response.text)


if __name__ == "__main__":
    main()