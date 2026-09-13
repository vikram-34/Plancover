"""Focused insurer and TPA identification for GMC policy opening pages."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Optional

from .ingest import IngestionResult, ingest_pdf
from .llm_client import GeminiLLMClient


IDENTITY_INSTRUCTIONS = (
    "Extract only the issuing insurance carrier and, if stated, the TPA from "
    "the policy opening pages. Do not infer names."
)


def build_insurer_tpa_prompt(opening_text: str) -> str:
    """Build a carrier-focused prompt that excludes people and intermediaries."""

    return f"""Identify ONLY the name of the insurance company/carrier that issued
this policy (for example, "TATA AIG General Insurance Co. Ltd." or
"ICICI Lombard General Insurance"). Do NOT return a person's name, broker
name, agent name, employer name, or policyholder name. The insurer name often
appears in the document header/letterhead or in the closing signature line
(e.g. "For [Company Name] Authorized Signatory"). Look in both locations,
as well as the policy text, even when there is no explicit "Insurer:" label.
If genuinely not found, return null for insurer.

Also identify the TPA (Third Party Administrator) name, if explicitly stated.
If a TPA is not mentioned, return null for tpa.

Return only this JSON object:
{{"insurer": null, "tpa": null}}

Policy opening pages:
{opening_text or "No extracted text was available."}
"""


def identify_policy_parties(document: IngestionResult, client: Any) -> dict[str, Optional[str]]:
    """Ask the LLM for the issuing carrier and TPA from opening and closing pages."""

    pages = document.pages[:3]
    if document.pages and document.pages[-1] not in pages:
        pages.append(document.pages[-1])
    opening_text = "\n\n".join(page.raw_text for page in pages)
    payload = client.complete_json(
        build_insurer_tpa_prompt(opening_text),
        instructions=IDENTITY_INSTRUCTIONS,
        schema={
            "type": "object",
            "properties": {
                "insurer": {"type": ["string", "null"]},
                "tpa": {"type": ["string", "null"]},
            },
            "additionalProperties": False,
        },
        max_attempts=1,
    )
    if not isinstance(payload, dict):
        raise ValueError("Insurer/TPA response must be a JSON object")
    return {
        "insurer": _identity_value(payload.get("insurer")),
        "tpa": _identity_value(payload.get("tpa")),
    }


def _identity_value(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    return (
        None
        if value.strip().casefold() in {"not found", "not mentioned", "unknown", "null", "none", "n/a"}
        else value
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Identify the issuing insurer and TPA for a policy PDF.")
    parser.add_argument("pdf_path", type=Path)
    args = parser.parse_args()

    client = GeminiLLMClient()
    result = identify_policy_parties(ingest_pdf(args.pdf_path), client)
    raw_response = getattr(client, "last_raw_response", None)
    if os.environ.get("DEBUG"):
        print(f"RAW RESPONSE [insurer_tpa]: {raw_response}")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
