"""Insurer-agnostic, chunked LLM extraction for GMC policy PDFs.

The extractor deliberately routes text by general insurance concepts rather than
by insurer.  Each extraction group sees only the chunks likely to contain its
terms, which keeps prompts focused and reduces token use while retaining page
fallbacks for policies whose headings are inconsistent or absent.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import date
from types import UnionType
from typing import Any, Iterable, Optional, Sequence
from typing import Union, get_args, get_origin, get_type_hints

from pydantic import ValidationError
from pydantic import BaseModel
from dateutil import parser as date_parser

from .detect_insurer import identify_policy_parties
from .ingest import IngestionResult, PageContent, ingest_pdf
from .llm_client import GeminiLLMClient
from .schema import GMCPPolicy


logger = logging.getLogger(__name__)

NOT_FOUND = "Not Found"
FULL_POLICY_INSTRUCTIONS = (
    "You extract structured data from a Group Medical Cover insurance policy. "
    "Return only the requested JSON object and never invent policy terms."
)
@dataclass(frozen=True)
class DocumentChunk:
    """A contiguous, page-aware portion of an ingested policy document."""

    page_numbers: tuple[int, ...]
    text: str
    detected_topics: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ExtractionGroup:
    """Fields and insurer-neutral terms used to locate one policy topic."""

    name: str
    fields: tuple[str, ...]
    keywords: tuple[str, ...]


@dataclass
class ExtractionResult:
    """Result of a full extraction, including non-fatal extraction diagnostics."""

    policy: Optional[GMCPPolicy]
    group_payloads: dict[str, dict[str, Any]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    alias_mappings: dict[str, dict[str, int]] = field(default_factory=dict)
    evidence: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


# These are conceptual insurance terms, not insurer-specific templates.  Related
# words make routing resilient to headings such as "Hospital Accommodation" or
# "Pre-existing Ailments" instead of one fixed policy vocabulary.
EXTRACTION_GROUPS: tuple[ExtractionGroup, ...] = (
    ExtractionGroup(
        name="policy_and_eligibility",
        fields=(
            "insurer_name",
            "tpa_name",
            "policy_period",
            "tenure",
            "policy_type",
            "total_premium",
            "total_sum_insured",
            "previous_year_premium",
            "family_structure",
            "sum_insured_tiers",
            "demographics_counts",
            "total_lives_covered",
            "hospitalization_periods",
        ),
        keywords=(
            "policy period",
            "policy tenure",
            "insured name",
            "third party administrator",
            "tpa",
            "sum insured",
            "coverage for",
            "eligible family",
            "employee spouse",
            "lives covered",
            "pre-hospitalization",
            "post-hospitalization",
        ),
    ),
    ExtractionGroup(
        name="room_rent_and_icu",
        fields=("room_rent", "icu_charges"),
        keywords=(
            "room rent",
            "room charges",
            "room accommodation",
            "hospital accommodation",
            "icu",
            "intensive care",
            "critical care",
            "per day",
        ),
    ),
    ExtractionGroup(
        name="maternity",
        fields=("maternity",),
        keywords=(
            "maternity",
            "normal delivery",
            "caesarean",
            "c-section",
            "new born",
            "newborn",
            "day one",
            "vaccination",
            "pregnancy",
        ),
    ),
    ExtractionGroup(
        name="waiting_periods",
        fields=("waiting_periods",),
        keywords=(
            "waiting period",
            "30 days",
            "thirty days",
            "first year",
            "second year",
            "pre-existing disease",
            "pre existing disease",
            "ped",
            "waiver",
        ),
    ),
    ExtractionGroup(
        name="other_benefits",
        fields=("other_benefits",),
        keywords=(
            "day care",
            "outpatient",
            "opd",
            "teleconsult",
            "pharmacy",
            "domiciliary",
            "health check",
            "modern treatment",
            "bariatric",
            "psychiatric",
            "ayush",
            "organ donor",
            "live-in",
            "lgbtq",
        ),
    ),
    ExtractionGroup(
        name="buffer_and_special_cover",
        fields=(
            "infertility_surrogacy",
            "ambulance",
            "air_ambulance",
            "corporate_buffer_limit",
            "disease_wise_capping",
        ),
        keywords=(
            "infertility",
            "surrogacy",
            "ambulance",
            "corporate buffer",
            "buffer limit",
            "disease-wise",
            "disease wise",
            "capping",
            "sub-limit",
        ),
    ),
)


DOCUMENT_GROUPS: tuple[ExtractionGroup, ...] = (
    ExtractionGroup(
        name="policy_core",
        fields=(
            "policy_period",
            "tenure",
            "policy_type",
            "total_premium",
            "total_sum_insured",
            "previous_year_premium",
            "family_structure",
            "sum_insured_tiers",
            "demographics_counts",
            "total_lives_covered",
        ),
        keywords=(),
    ),
    ExtractionGroup(
        name="coverage_and_waiting",
        fields=(
            "room_rent",
            "icu_charges",
            "hospitalization_periods",
            "maternity",
            "waiting_periods",
        ),
        keywords=(),
    ),
    ExtractionGroup(
        name="benefits_and_limits",
        fields=(
            "other_benefits",
            "infertility_surrogacy",
            "ambulance",
            "air_ambulance",
            "corporate_buffer_limit",
            "disease_wise_capping",
        ),
        keywords=(),
    ),
)


class GMCExtractor:
    """Extract a validated GMC policy record using targeted LLM calls."""

    def __init__(
        self,
        *,
        client: Any | None = None,
        model: str | None = None,
        pages_per_fallback_chunk: int = 2,
        max_prompt_characters: int = 30_000,
    ) -> None:
        self._client = client
        # Passing None lets GeminiLLMClient read GEMINI_MODEL from .env.
        self.model = model
        self.pages_per_fallback_chunk = pages_per_fallback_chunk
        self.max_prompt_characters = max_prompt_characters

    def extract_pdf(self, pdf_path: str) -> ExtractionResult:
        """Ingest a PDF then run group-wise extraction over its content."""

        return self.extract(ingest_pdf(pdf_path))

    def extract(self, document: IngestionResult) -> ExtractionResult:
        """Extract a policy through three smaller, schema-scoped LLM calls."""

        errors: list[str] = []
        alias_mappings: dict[str, dict[str, int]] = {}
        group_payloads: dict[str, dict[str, Any]] = {}
        merged: dict[str, Any] = {}
        document_text = render_full_document(document)
        client = self._get_client()

        for group in DOCUMENT_GROUPS:
            skeleton = group_skeleton(group.fields)
            prompt = build_group_document_prompt(group, document_text, skeleton)
            try:
                payload = client.complete_json(
                    prompt,
                    instructions=FULL_POLICY_INSTRUCTIONS,
                    schema=group_json_schema(group),
                    max_attempts=2,
                )
                raw_response = getattr(client, "last_raw_response", None)
                _print_raw_response(group.name, raw_response, payload)
                if not isinstance(payload, dict):
                    raise ValueError("LLM response must be a JSON object")
                payload = unwrap_group_payload(payload, group)
                payload = normalize_group_keys(payload, group.fields, alias_mappings)
                payload = normalize_payload_values(payload, group.fields, alias_mappings)
                payload = normalize_not_found(payload)
                unexpected_fields = set(payload) - set(group.fields)
                if unexpected_fields:
                    logger.warning(
                        "Ignoring unexpected fields in %s response: %s",
                        group.name,
                        sorted(unexpected_fields),
                    )
                    payload = {key: value for key, value in payload.items() if key in group.fields}
                group_payloads[group.name] = payload
                merged.update(payload)
                missing_fields = _missing_group_fields(payload, group.fields)
                if missing_fields:
                    retry_prompt = build_missing_fields_prompt(
                        group, document_text, missing_fields
                    )
                    retry_payload = client.complete_json(
                        retry_prompt,
                        instructions=FULL_POLICY_INSTRUCTIONS,
                        schema=group_json_schema(group),
                        max_attempts=2,
                    )
                    if not isinstance(retry_payload, dict):
                        raise ValueError("LLM retry response must be a JSON object")
                    retry_payload = unwrap_group_payload(retry_payload, group)
                    retry_payload = normalize_group_keys(
                        retry_payload, group.fields, alias_mappings
                    )
                    retry_payload = normalize_payload_values(
                        retry_payload, group.fields, alias_mappings
                    )
                    retry_payload = normalize_not_found(retry_payload)
                    for field_name in missing_fields:
                        retry_value = retry_payload.get(field_name)
                        if _has_meaningful_value(retry_value):
                            payload[field_name] = retry_value
                            merged[field_name] = retry_value
                    group_payloads[group.name] = payload
            except Exception as exc:
                message = f"{group.name} extraction failed: {exc}"
                logger.warning(message)
                errors.append(message)

        _apply_schedule_evidence(merged, document)

        try:
            policy = GMCPPolicy.model_validate(merged)
        except ValidationError as exc:
            message = f"Final GMC policy validation failed: {exc}"
            logger.error(message)
            errors.append(message)
            policy = None

        warnings: list[str] = []
        if policy is not None:
            populated, total = populated_top_level_fields(policy)
            message = f"Only {populated}/{total} top-level fields populated"
            logger.warning(message)
            warnings.append(message)

        evidence = build_field_evidence(policy or GMCPPolicy(), document)

        return ExtractionResult(
            policy=policy,
            group_payloads=group_payloads,
            errors=errors,
            warnings=warnings,
            alias_mappings=alias_mappings,
            evidence=evidence,
        )

    def identify_insurer_tpa(self, document: IngestionResult) -> dict[str, Optional[str]]:
        """Use a small LLM call over the opening pages to identify insurer and TPA."""

        client = self._get_client()
        parties = identify_policy_parties(document, client)
        raw_response = getattr(client, "last_raw_response", None)
        _print_raw_response("insurer_tpa", raw_response, parties)
        return parties

    def _extract_group(
        self,
        group: ExtractionGroup,
        chunks: Sequence[DocumentChunk],
        alias_mappings: dict[str, dict[str, int]],
    ) -> dict[str, Any]:
        client = self._get_client()

        schema = group_json_schema(group)
        prompt = build_extraction_prompt(group, chunks, schema, self.max_prompt_characters)
        payload = client.complete_json(
            prompt,
            instructions=(
                "You extract information from Group Medical Cover insurance policies. "
                "Return only the requested JSON object; never invent policy terms."
            ),
            schema=schema,
        )
        if not isinstance(payload, dict):
            raise ValueError("The LLM response must be a JSON object")
        payload = normalize_group_keys(payload, group.fields, alias_mappings)
        payload = normalize_payload_values(payload, group.fields, alias_mappings)
        unexpected_fields = set(payload) - set(group.fields)
        if unexpected_fields:
            logger.warning(
                "Ignoring unexpected fields in %s group response: %s",
                group.name,
                sorted(unexpected_fields),
            )
            payload = {key: value for key, value in payload.items() if key in group.fields}
        return payload

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        self._client = GeminiLLMClient(model=self.model)
        return self._client


def policy_skeleton() -> dict[str, Any]:
    """Return the complete Pydantic-shaped output skeleton for one policy."""

    return GMCPPolicy().model_dump(mode="json")


def group_skeleton(fields: Sequence[str]) -> dict[str, Any]:
    """Return only the requested top-level fields from the full policy skeleton."""

    full_skeleton = policy_skeleton()
    return {field_name: full_skeleton[field_name] for field_name in fields}


def _missing_group_fields(payload: dict[str, Any], fields: Sequence[str]) -> list[str]:
    """Return fields that are absent or still contain only schema defaults."""

    defaults = group_skeleton(fields)
    return [
        field_name
        for field_name in fields
        if not _has_meaningful_value(payload.get(field_name, defaults[field_name]))
    ]


def build_missing_fields_prompt(
    group: ExtractionGroup, document_text: str, missing_fields: Sequence[str]
) -> str:
    """Ask a second pass to recover only fields left empty by the first pass."""

    skeleton = group_skeleton(missing_fields)
    return f"""Recheck this GMC policy only for these missing fields: {list(missing_fields)}.

Return one JSON object containing exactly those fields and the same nested shape
shown below. Search every page carefully, including tables, footnotes, exclusions,
and schedule sections. Use only direct evidence. Use \"{NOT_FOUND}\" when the
field is genuinely absent; never guess or overwrite source wording.

Exact JSON skeleton:
{json.dumps(skeleton, ensure_ascii=False, indent=2)}

Complete extracted document text:
{document_text or "No extracted text or tables were available."}
"""


def unwrap_group_payload(payload: dict[str, Any], group: ExtractionGroup) -> dict[str, Any]:
    """Unwrap the harmless group-name envelope often added by local models."""

    if set(payload) == {group.name} and isinstance(payload[group.name], dict):
        logger.warning("Unwrapped local-model %s response envelope.", group.name)
        return payload[group.name]

    unwrapped = dict(payload)
    for field_name in group.fields:
        field_value = unwrapped.get(field_name)
        if isinstance(field_value, dict) and set(field_value) == {group.name}:
            logger.warning("Unwrapped %s envelope inside %s.", group.name, field_name)
            unwrapped[field_name] = field_value[group.name]
    return unwrapped


def render_full_document(document: IngestionResult) -> str:
    """Render all extracted pages and tables without section filtering."""

    return "\n\n".join(_render_page(page) for page in document.pages)


_SCHEDULE_DATE_PATTERNS: dict[str, tuple[str, ...]] = {
    "start_date": (
        "date and time of policy commencement",
        "commencement date",
        "policy start date",
        "policy period start date",
    ),
    "end_date": (
        "date and time of policy expiry",
        "expiry date",
        "policy end date",
        "policy period end date",
    ),
}
_POLICY_DATE_PATTERN = re.compile(
    r"\b(?:\d{1,2}[-/](?:\d{1,2}|[A-Za-z]{3,9})[-/]\d{4}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})\b"
)
_AGGREGATE_SUM_INSURED_PATTERN = re.compile(r"aggregate\s+sum\s+insured", re.IGNORECASE)
_MONEY_CANDIDATE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:INR|Rs\.?\s*)?(\d{1,3}(?:,\d{2,3})+|\d{5,}(?:\.\d+)?)"
)


def _apply_schedule_evidence(payload: dict[str, Any], document: IngestionResult) -> None:
    """Prefer explicitly labelled schedule values over ambiguous model guesses.

    These generic labels occur across insurers and are extracted only when the
    PDF text directly associates a nearby value with the label. This avoids
    confusing an aggregate SI with an age-band or premium-rate figure.
    """

    period = payload.get("policy_period")
    if not isinstance(period, dict):
        period = {}
        payload["policy_period"] = period

    for field_name, labels in _SCHEDULE_DATE_PATTERNS.items():
        date_value = _find_schedule_date(document.pages, labels)
        if date_value:
            if period.get(field_name) != date_value:
                logger.info("Using labelled policy-schedule %s: %s", field_name, date_value)
            period[field_name] = date_value

    aggregate_sum_insured = _find_aggregate_sum_insured(document.pages)
    if aggregate_sum_insured:
        if payload.get("total_sum_insured") != aggregate_sum_insured:
            logger.info(
                "Using labelled Aggregate Sum Insured instead of model-selected value: %s",
                aggregate_sum_insured,
            )
        payload["total_sum_insured"] = aggregate_sum_insured


def _find_schedule_date(pages: Sequence[PageContent], labels: Sequence[str]) -> str | None:
    for page in pages:
        lines = [line.strip() for line in page.raw_text.splitlines() if line.strip()]
        for index, line in enumerate(lines):
            if not any(label in line.casefold() for label in labels):
                continue
            for candidate_line in lines[index + 1 : index + 6]:
                match = _POLICY_DATE_PATTERN.search(candidate_line)
                if match:
                    parsed = _parse_policy_date(match.group(0), "policy_schedule")
                    if parsed:
                        return parsed
    return None


def _find_aggregate_sum_insured(pages: Sequence[PageContent]) -> str | None:
    for page in pages:
        for label_match in _AGGREGATE_SUM_INSURED_PATTERN.finditer(page.raw_text):
            nearby_text = page.raw_text[label_match.end() : label_match.end() + 2_000]
            candidates: list[tuple[float, str]] = []
            for money_match in _MONEY_CANDIDATE_PATTERN.finditer(nearby_text):
                candidate = money_match.group(1)
                numeric_value = float(candidate.replace(",", ""))
                if 100_000 <= numeric_value <= 1_000_000_000:
                    candidates.append((numeric_value, candidate))
            if candidates:
                return max(candidates, key=lambda item: item[0])[1]
    return None


def build_group_document_prompt(
    group: ExtractionGroup, document_text: str, skeleton: dict[str, Any]
) -> str:
    """Build a compact full-document prompt for one logical schema slice."""

    sum_insured_instruction = ""
    if "total_sum_insured" in group.fields:
        sum_insured_instruction = (
            "\nFor `total_sum_insured`, extract only the Aggregate Sum Insured or "
            "Total Sum Insured stated in the policy schedule. Do NOT use a figure "
            "from a premium-rate table, premium-rater table, age-band table, or "
            "individual rate band.\n"
        )

    return f"""Extract only the `{group.name}` section from this GMC policy.

Copy this exact JSON structure. Replace only placeholder values with policy data.
Keep every key exactly as shown, including nested keys. Do not rename, add,
remove, or flatten keys. Coverage and family-relation values in this skeleton
are flat strings: write one concise source-faithful description containing the
coverage status, limit, and conditions together. Use "Not Found" only when no
direct evidence exists. Preserve monetary limits, percentages, and day counts
verbatim.
{sum_insured_instruction}

Exact JSON skeleton:
{json.dumps(skeleton, ensure_ascii=False, indent=2)}

Complete extracted document text:
{document_text or "No extracted text or tables were available."}
"""


def populated_top_level_fields(policy: GMCPPolicy) -> tuple[int, int]:
    """Count top-level fields that contain a meaningful, non-default value."""

    payload = policy.model_dump(mode="json")
    payload.pop("extraction_status", None)
    payload.pop("extraction_evidence", None)
    populated = sum(_has_meaningful_value(value) for value in payload.values())
    return populated, len(payload)


def _has_meaningful_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        return any(_has_meaningful_value(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_meaningful_value(item) for item in value)
    if isinstance(value, str):
        return bool(value.strip()) and value.strip().casefold() != NOT_FOUND.casefold()
    return True


def build_field_evidence(
    policy: GMCPPolicy, document: IngestionResult, *, snippet_characters: int = 220
) -> dict[str, list[dict[str, Any]]]:
    """Attach nearby source snippets to populated fields for human verification."""

    evidence: dict[str, list[dict[str, Any]]] = {}
    policy_payload = policy.model_dump(mode="json")
    for field_name, value in policy_payload.items():
        if field_name in {"extraction_status", "extraction_evidence"} or not _has_meaningful_value(value):
            continue
        terms = (field_name.replace("_", " "),) + EXPLICIT_ALIASES.get(field_name, ())
        matches: list[dict[str, Any]] = []
        for page in document.pages:
            lower_text = page.raw_text.casefold()
            for term in terms:
                position = lower_text.find(term.casefold())
                if position < 0:
                    continue
                start = max(0, position - snippet_characters // 2)
                end = min(len(page.raw_text), position + len(term) + snippet_characters // 2)
                matches.append({"page": page.page_number, "snippet": page.raw_text[start:end].strip()})
                break
        if matches:
            evidence[field_name] = matches[:3]
    return evidence


def build_full_policy_prompt(document_text: str, skeleton: dict[str, Any]) -> str:
    """Build the one-call extraction prompt with the complete JSON structure."""

    skeleton_json = json.dumps(skeleton, ensure_ascii=False, indent=2)
    return f"""Extract the complete GMC policy record from the document below.

Copy this exact JSON structure. Replace only the placeholder values with data
found in the text below. Keep every key exactly as shown, including nested keys.
Do not rename, add, remove, or flatten any keys.

When a value is not present, keep it null (or keep an empty list where the
skeleton uses an empty list). Do not guess. Preserve monetary limits,
percentages, dates, and day counts from the source.

Exact JSON skeleton:
{skeleton_json}

Complete extracted document text:
{document_text or "No extracted text or tables were available."}
"""


def build_repair_prompt(
    document_text: str, skeleton: dict[str, Any], raw_output: str
) -> str:
    """Build the single repair prompt for malformed or invalid model output."""

    return f"""Fix this extracted GMC policy JSON to match the exact structure.

Copy this exact JSON structure. Keep every key exactly as shown, including nested
keys. Do not rename, add, remove, or flatten any keys. Preserve any correct
values from the model output, and use null or an empty list when the document
does not provide a value. Return only JSON.

Exact JSON skeleton:
{json.dumps(skeleton, ensure_ascii=False, indent=2)}

Model output to repair:
{raw_output}

Source document text:
{document_text or "No extracted text or tables were available."}
"""


def normalize_and_validate_policy(
    payload: dict[str, Any], alias_mappings: dict[str, dict[str, int]]
) -> Optional[GMCPPolicy]:
    """Apply key/value normalisation once, then return a validated policy."""

    expected_fields = tuple(GMCPPolicy.model_fields)
    normalized = normalize_group_keys(payload, expected_fields, alias_mappings)
    normalized = normalize_payload_values(normalized, expected_fields, alias_mappings)
    normalized = normalize_not_found(normalized)
    unexpected_fields = set(normalized) - set(expected_fields)
    if unexpected_fields:
        logger.warning(
            "Ignoring unexpected fields in document response: %s",
            sorted(unexpected_fields),
        )
        normalized = {key: value for key, value in normalized.items() if key in expected_fields}
    try:
        return GMCPPolicy.model_validate(normalized)
    except ValidationError as exc:
        logger.error("Final GMC policy validation failed: %s", exc)
        return None


def chunk_document(document: IngestionResult, max_pages_per_chunk: int = 2) -> list[DocumentChunk]:
    """Create contiguous logical chunks, falling back to fixed page groups.

    A page containing a likely section heading starts a new chunk when it changes
    topic.  Pages without a recognisable heading are still grouped by page count,
    making the method useful for unstructured and insurer-specific layouts.
    """

    if max_pages_per_chunk < 1:
        raise ValueError("max_pages_per_chunk must be at least 1")

    chunks: list[DocumentChunk] = []
    current_pages: list[PageContent] = []
    current_topics: set[str] = set()

    for page in document.pages:
        page_topics = detect_topics(page.raw_text)
        topic_change = bool(current_pages and page_topics and not page_topics.issubset(current_topics))
        at_page_limit = len(current_pages) >= max_pages_per_chunk
        if current_pages and (topic_change or at_page_limit):
            chunks.append(_make_chunk(current_pages, current_topics))
            current_pages = []
            current_topics = set()

        current_pages.append(page)
        current_topics.update(page_topics)

    if current_pages:
        chunks.append(_make_chunk(current_pages, current_topics))
    return chunks


def select_relevant_chunks(
    group: ExtractionGroup,
    chunks: Sequence[DocumentChunk],
    pages: Sequence[PageContent],
) -> list[DocumentChunk]:
    """Return keyword-matched chunks, or one/two representative fallback pages."""

    matched = [chunk for chunk in chunks if _contains_any(chunk.text, group.keywords)]
    if matched:
        return matched

    logger.info("No matching chunk for %s; using fallback pages.", group.name)
    if not pages:
        return []
    fallback_pages = [pages[0]]
    if len(pages) > 1:
        fallback_pages.append(pages[-1])
    return [_make_chunk([page], detect_topics(page.raw_text)) for page in fallback_pages]


def group_json_schema(group: ExtractionGroup) -> dict[str, Any]:
    """Build a group-only JSON Schema from the canonical Pydantic model."""

    full_schema = GMCPPolicy.model_json_schema()
    properties = full_schema["properties"]
    return {
        "type": "object",
        "properties": {name: properties[name] for name in group.fields},
        "additionalProperties": False,
        "$defs": full_schema.get("$defs", {}),
    }


def build_extraction_prompt(
    group: ExtractionGroup,
    chunks: Sequence[DocumentChunk],
    schema: dict[str, Any],
    max_characters: int,
) -> str:
    """Build a scoped, evidence-preserving instruction for one field group."""

    evidence = _render_chunks(chunks, max_characters)
    exact_keys = json.dumps(list(group.fields), ensure_ascii=False)
    return f"""Extract only the `{group.name}` field group from this GMC policy.

Return one JSON object with these exact top-level keys: {exact_keys}.
Return ONLY these exact keys: {exact_keys}. Do not rename, add, or omit keys.
Follow the supplied Pydantic-derived JSON shape exactly. Do not include markdown,
explanations, or fields from another group.

Evidence rules:
- Use only the supplied policy pages and tables; never infer, calculate, or guess.
- When information is absent, write exactly \"{NOT_FOUND}\" in that value rather
  than guessing.
- Preserve currency amounts, monetary limits, percentages, and day counts verbatim
  from the source; do not paraphrase or convert them.
- Coverage and family-relation fields are single strings. Combine status, limit,
  and conditions in source-faithful wording; do not create nested objects.

Pydantic-derived JSON schema:
{json.dumps(schema, ensure_ascii=False)}

Relevant policy evidence:
{evidence}
"""


def normalize_not_found(value: Any) -> Any:
    """Retain the explicit no-evidence sentinel used by flat coverage fields."""

    return value


EXPLICIT_ALIASES: dict[str, tuple[str, ...]] = {
    "total_premium": (
        "premium",
        "total premium",
        "premium amount",
        "annual premium",
        "gross premium",
    ),
    "total_sum_insured": (
        "total sum insured",
        "total sum inured",
        "overall sum insured",
        "sum insured",
    ),
    "policy_type": ("cover type", "coverage type", "type of policy"),
    "insurer_name": ("insurer", "insurance company", "insurance provider", "underwriter"),
    "tpa_name": ("tpa", "third party administrator", "third-party administrator"),
    "previous_year_premium": ("last year premium", "prior year premium", "previous premium"),
    "ambulance": ("ambulance charges", "road ambulance", "ground ambulance"),
    "air_ambulance": ("air ambulance charges", "air ambulance cover"),
    "room_rent": ("room rent charges", "room charges", "hospital accommodation"),
    "icu_charges": ("icu charges", "intensive care charges", "critical care charges"),
    "maternity": ("maternity benefits", "maternity cover"),
    "waiting_periods": ("waiting period details", "waiting period provisions"),
    "other_benefits": ("other benefits", "additional benefits", "ancillary benefits"),
    "modern_treatment": ("modern treatments", "modern treatment coverage"),
    "infertility_surrogacy": ("infertility and surrogacy", "surrogacy cover"),
    "corporate_buffer_limit": ("corporate buffer", "corporate floater sum insured", "buffer limit"),
    "disease_wise_capping": ("disease wise cap", "disease-wise cap", "disease capping"),
}


def normalize_group_keys(
    payload: dict[str, Any],
    expected_fields: Sequence[str],
    alias_mappings: dict[str, dict[str, int]] | None = None,
) -> dict[str, Any]:
    """Map unambiguous model key variants to canonical Pydantic field names.

    Normalisation is deliberately conservative: case, whitespace, underscores,
    and hyphens are interchangeable, while an unknown key is retained so the
    caller can warn and discard it.  A common flattened form such as
    ``Policy Period - Start Date`` is also safely converted to
    ``{"policy_period": {"start_date": ...}}``.
    """

    expected = tuple(expected_fields)
    canonical_to_expected = build_alias_map(expected)
    normalized: dict[str, Any] = {}

    for raw_key, value in payload.items():
        raw_key_text = str(raw_key)
        direct_key = canonical_to_expected.get(_canonical_key(raw_key_text))
        if direct_key:
            if alias_mappings is not None and raw_key_text != direct_key:
                _record_alias(alias_mappings, raw_key_text, direct_key)
            normalized[direct_key] = _normalize_nested_keys(direct_key, value, alias_mappings)
            continue

        flattened_match = _match_flattened_nested_key(str(raw_key), expected)
        if flattened_match:
            parent_key, child_key = flattened_match
            parent_value = normalized.setdefault(parent_key, {})
            if isinstance(parent_value, dict):
                parent_value[child_key] = value
                if alias_mappings is not None:
                    _record_alias(alias_mappings, raw_key_text, f"{parent_key}.{child_key}")
            else:
                logger.warning("Could not merge flattened key %r into %s", raw_key, parent_key)
            continue

        normalized[str(raw_key)] = value

    return normalized


def build_alias_map(expected_fields: Sequence[str]) -> dict[str, str]:
    """Build canonical aliases from field names plus safe explicit synonyms."""

    alias_map: dict[str, str] = {}
    for field_name in expected_fields:
        alias_map[_canonical_key(field_name)] = field_name
    for target, aliases in EXPLICIT_ALIASES.items():
        if target not in expected_fields:
            continue
        for alias in aliases:
            canonical_alias = _canonical_key(alias)
            existing = alias_map.get(canonical_alias)
            if existing is None or existing == target:
                alias_map[canonical_alias] = target
    return alias_map


def _record_alias(usage: dict[str, dict[str, int]], source: str, target: str) -> None:
    source_usage = usage.setdefault(source, {})
    source_usage[target] = source_usage.get(target, 0) + 1


def normalize_payload_values(
    payload: dict[str, Any],
    expected_fields: Sequence[str],
    alias_mappings: dict[str, dict[str, int]] | None = None,
) -> dict[str, Any]:
    """Normalise dates and coerce flat text values before validation."""

    model_hints = get_type_hints(GMCPPolicy)
    normalized: dict[str, Any] = {}
    for field_name, value in payload.items():
        annotation = model_hints.get(field_name)
        normalized[field_name] = _normalize_value(value, annotation, field_name, alias_mappings)
    return normalized


def _normalize_value(
    value: Any,
    annotation: Any,
    path: str,
    alias_mappings: dict[str, dict[str, int]] | None = None,
) -> Any:
    if annotation is None:
        return value

    annotation = _unwrap_optional(annotation)
    if annotation is None:
        return value

    if annotation is str:
        if value is None:
            return NOT_FOUND
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, dict):
            extracted = _first_string_value(value)
            if extracted is not None:
                logger.warning(
                    "Coerced nested object to flat string for %s using value %r", path, extracted
                )
                return extracted
            logger.warning("Coerced nested object to JSON string for %s", path)
            return json.dumps(value, ensure_ascii=False, default=str)
        if isinstance(value, list):
            return "; ".join(str(item) for item in value)
        return value

    if annotation is date and isinstance(value, str):
        parsed_date = _parse_policy_date(value, path)
        return parsed_date or value

    if isinstance(value, dict) and _is_model(annotation):
        return _normalize_model_dict(value, annotation, path, alias_mappings)

    if isinstance(value, list) and get_origin(annotation) is list:
        item_annotation = get_args(annotation)[0] if get_args(annotation) else None
        return [
            _normalize_value(item, item_annotation, f"{path}[{index}]", alias_mappings)
            for index, item in enumerate(value)
        ]

    return value


def _first_string_value(value: Any) -> str | None:
    """Find the first non-empty string in a nested local-model response."""

    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, dict):
        for nested_value in value.values():
            found = _first_string_value(nested_value)
            if found is not None:
                return found
    elif isinstance(value, list):
        for nested_value in value:
            found = _first_string_value(nested_value)
            if found is not None:
                return found
    return None


def _normalize_model_dict(
    value: dict[str, Any],
    model_type: type[BaseModel],
    path: str,
    alias_mappings: dict[str, dict[str, int]] | None = None,
) -> dict[str, Any]:
    """Normalize nested model keys and values using that model's type hints."""

    normalized_keys = normalize_group_keys(value, tuple(model_type.model_fields), alias_mappings)
    hints = get_type_hints(model_type)
    unexpected_fields = set(normalized_keys) - set(model_type.model_fields)
    if unexpected_fields:
        logger.warning(
            "Ignoring unexpected nested fields in %s: %s",
            path,
            sorted(unexpected_fields),
        )
    return {
        key: _normalize_value(item, hints.get(key), f"{path}.{key}", alias_mappings)
        for key, item in normalized_keys.items()
        if key in model_type.model_fields
    }


def _is_model(annotation: Any) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


def _unwrap_optional(annotation: Any) -> Any:
    if get_origin(annotation) in (Union, UnionType):
        return next((arg for arg in get_args(annotation) if arg is not type(None)), None)
    return annotation


def _parse_policy_date(value: str, path: str) -> str | None:
    """Strip common time fragments and return an ISO date when parseable."""

    cleaned = re.sub(r"\bmidnight\b", " ", value, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:hrs?|hours?)?\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:AM|PM)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;:-")
    try:
        return date_parser.parse(cleaned, fuzzy=True, dayfirst=True).date().isoformat()
    except (ValueError, OverflowError, TypeError) as exc:
        logger.warning("Could not parse date at %s (%r): %s", path, value, exc)
        return None


def _normalize_nested_keys(
    field_name: str,
    value: Any,
    alias_mappings: dict[str, dict[str, int]] | None = None,
) -> Any:
    if not isinstance(value, dict):
        return value
    model_type = _nested_model_type(field_name)
    if model_type is None:
        return {
            _freeform_key(str(key)): item
            for key, item in value.items()
        }
    expected = tuple(model_type.model_fields)
    return normalize_group_keys(value, expected, alias_mappings)


def _match_flattened_nested_key(raw_key: str, expected_fields: Sequence[str]) -> tuple[str, str] | None:
    parts = re.split(r"\s+(?:-|–|:)\s+|\s+-\s*|\s+–\s*", raw_key.strip(), maxsplit=1)
    if len(parts) != 2:
        canonical_raw = _canonical_key(raw_key)
        prefix_matches = []
        for field_name in expected_fields:
            parent_canonical = _canonical_key(field_name)
            if not canonical_raw.startswith(parent_canonical) or canonical_raw == parent_canonical:
                continue
            model_type = _nested_model_type(field_name)
            if model_type is None:
                continue
            remainder = canonical_raw[len(parent_canonical):]
            child_matches = [
                child_name
                for child_name in model_type.model_fields
                if remainder == _canonical_key(child_name)
            ]
            if len(child_matches) == 1:
                prefix_matches.append((field_name, child_matches[0]))
        return prefix_matches[0] if len(prefix_matches) == 1 else None
    parent_matches = [
        field_name
        for field_name in expected_fields
        if _canonical_key(parts[0]) == _canonical_key(field_name)
    ]
    if len(parent_matches) != 1:
        return None
    model_type = _nested_model_type(parent_matches[0])
    if model_type is None:
        return None
    child_matches = [
        child_name
        for child_name in model_type.model_fields
        if _canonical_key(parts[1]) == _canonical_key(child_name)
    ]
    return (parent_matches[0], child_matches[0]) if len(child_matches) == 1 else None


def _nested_model_type(field_name: str) -> type[BaseModel] | None:
    annotation = get_type_hints(GMCPPolicy).get(field_name)
    if annotation is None:
        return None
    if get_origin(annotation) in (Union, UnionType):
        annotation = next((arg for arg in get_args(annotation) if arg is not type(None)), None)
    return annotation if isinstance(annotation, type) and issubclass(annotation, BaseModel) else None


def _canonical_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _freeform_key(value: str) -> str:
    """Normalize an untyped nested key while retaining readable separators."""

    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def detect_topics(text: str) -> set[str]:
    """Detect broad insurance concepts without relying on insurer-specific labels."""

    lower_text = text.casefold()
    return {
        group.name
        for group in EXTRACTION_GROUPS
        if _contains_any(lower_text, group.keywords)
    }


def _make_chunk(pages: Iterable[PageContent], topics: Iterable[str]) -> DocumentChunk:
    page_list = list(pages)
    return DocumentChunk(
        page_numbers=tuple(page.page_number for page in page_list),
        text="\n\n".join(_render_page(page) for page in page_list),
        detected_topics=frozenset(topics),
    )


def _render_page(page: PageContent) -> str:
    sections = [f"--- PAGE {page.page_number} ---", page.raw_text]
    if page.tables:
        sections.append("TABLES:\n" + json.dumps(page.tables, ensure_ascii=False))
    return "\n".join(section for section in sections if section)


def _render_chunks(chunks: Sequence[DocumentChunk], max_characters: int) -> str:
    if max_characters < 1:
        raise ValueError("max_prompt_characters must be at least 1")
    rendered: list[str] = []
    remaining = max_characters
    for chunk in chunks:
        if remaining <= 0:
            logger.warning("Relevant evidence truncated at %s characters.", max_characters)
            break
        text = chunk.text[:remaining]
        rendered.append(text)
        remaining -= len(text)
    return "\n\n".join(rendered) or "No extracted text or tables were available."


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    return any(re.search(rf"(?<!\w){re.escape(keyword.casefold())}(?!\w)", text.casefold()) for keyword in keywords)


def _print_raw_response(group_name: str, raw_response: str | None, payload: Any) -> None:
    """Emit sensitive model output only when explicitly debugging locally."""

    if os.environ.get("DEBUG"):
        print(
            f"RAW RESPONSE [{group_name}]: "
            + (raw_response or json.dumps(payload, ensure_ascii=False, default=str))
        )
