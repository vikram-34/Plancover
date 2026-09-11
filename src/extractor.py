"""Insurer-agnostic, chunked LLM extraction for GMC policy PDFs.

The extractor deliberately routes text by general insurance concepts rather than
by insurer.  Each extraction group sees only the chunks likely to contain its
terms, which keeps prompts focused and reduces token use while retaining page
fallbacks for policies whose headings are inconsistent or absent.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date
from types import UnionType
from typing import Any, Iterable, Optional, Sequence
from typing import Union, get_args, get_origin, get_type_hints

from pydantic import ValidationError
from pydantic import BaseModel
from dateutil import parser as date_parser

from .ingest import IngestionResult, PageContent, ingest_pdf
from .llm_client import OllamaLLMClient
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
    alias_mappings: dict[str, dict[str, int]] = field(default_factory=dict)


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
        # Passing None lets OllamaLLMClient read OLLAMA_MODEL from .env.
        self.model = model
        self.pages_per_fallback_chunk = pages_per_fallback_chunk
        self.max_prompt_characters = max_prompt_characters

    def extract_pdf(self, pdf_path: str) -> ExtractionResult:
        """Ingest a PDF then run group-wise extraction over its content."""

        return self.extract(ingest_pdf(pdf_path))

    def extract(self, document: IngestionResult) -> ExtractionResult:
        """Extract one complete policy in one call, with one optional repair call."""

        errors: list[str] = []
        alias_mappings: dict[str, dict[str, int]] = {}
        skeleton = policy_skeleton()
        document_text = render_full_document(document)
        client = self._get_client()
        prompt = build_full_policy_prompt(document_text, skeleton)

        try:
            raw_payload = client.complete_json(
                prompt,
                instructions=FULL_POLICY_INSTRUCTIONS,
                schema=GMCPPolicy.model_json_schema(),
                max_attempts=1,
            )
        except Exception as exc:
            raw_payload = None
            errors.append(f"Initial document extraction failed: {exc}")
            logger.warning(errors[-1])

        policy = None
        if isinstance(raw_payload, dict):
            policy = normalize_and_validate_policy(raw_payload, alias_mappings)
            if policy is None:
                errors.append("Initial document response failed policy validation")

        if policy is None:
            raw_output = getattr(client, "last_raw_response", None)
            if raw_output is None:
                raw_output = json.dumps(raw_payload, ensure_ascii=False, default=str)
            repair_prompt = build_repair_prompt(document_text, skeleton, raw_output)
            try:
                repaired_payload = client.complete_json(
                    repair_prompt,
                    instructions=FULL_POLICY_INSTRUCTIONS,
                    schema=GMCPPolicy.model_json_schema(),
                    max_attempts=1,
                )
                policy = normalize_and_validate_policy(repaired_payload, alias_mappings)
                if policy is None:
                    errors.append("Repair response failed policy validation")
            except Exception as exc:
                errors.append(f"Repair extraction failed: {exc}")
                logger.warning(errors[-1])

        if policy is None:
            logger.error("Final GMC policy validation failed after initial and repair attempts")
        return ExtractionResult(
            policy=policy,
            group_payloads={"document": raw_payload} if isinstance(raw_payload, dict) else {},
            errors=errors,
            alias_mappings=alias_mappings,
        )

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
        self._client = OllamaLLMClient(model=self.model)
        return self._client


def policy_skeleton() -> dict[str, Any]:
    """Return the complete Pydantic-shaped output skeleton for one policy."""

    return GMCPPolicy().model_dump(mode="json")


def render_full_document(document: IngestionResult) -> str:
    """Render all extracted pages and tables without section filtering."""

    return "\n\n".join(_render_page(page) for page in document.pages)


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
  than guessing. The calling code will normalise this sentinel before validation.
- Preserve currency amounts, monetary limits, percentages, and day counts verbatim
  from the source; do not paraphrase or convert them.
- For coverage benefits, preserve `status`, `limit`, and `notes`. Use the exact
  allowed status wording when the policy establishes it.

Pydantic-derived JSON schema:
{json.dumps(schema, ensure_ascii=False)}

Relevant policy evidence:
{evidence}
"""


def normalize_not_found(value: Any) -> Any:
    """Convert the explicit no-evidence sentinel to ``None`` for Pydantic validation."""

    if isinstance(value, str) and value.strip().casefold() == NOT_FOUND.casefold():
        return None
    if isinstance(value, list):
        return [normalize_not_found(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize_not_found(item) for key, item in value.items()}
    return value


EXPLICIT_ALIASES: dict[str, tuple[str, ...]] = {
    "total_premium": ("premium", "total premium", "premium amount", "annual premium"),
    "total_sum_insured": ("total sum insured", "total sum inured", "overall sum insured"),
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
    "limit": ("amount", "coverage amount", "maximum amount"),
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
    """Normalise dates, enum synonyms, and benefit shorthand before validation."""

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

    if annotation is str and isinstance(value, (int, float)):
        return str(value)
    if annotation is str and isinstance(value, list):
        return "; ".join(str(item) for item in value)

    if _is_benefit_model(annotation) and isinstance(value, str):
        logger.warning(
            "Wrapped bare benefit string at %s as Covered with the string as its limit.",
            path,
        )
        return {"status": "Covered", "limit": value, "notes": None}

    if annotation is date and isinstance(value, str):
        parsed_date = _parse_policy_date(value, path)
        return parsed_date or value

    if _is_waiting_status(annotation) and isinstance(value, str):
        mapped_status = _map_waiting_status(value)
        if mapped_status != value:
            logger.warning("Mapped waiting-period status at %s: %r -> %r", path, value, mapped_status)
        return mapped_status

    if isinstance(value, dict) and _is_model(annotation):
        return _normalize_model_dict(value, annotation, path, alias_mappings)

    if isinstance(value, list) and get_origin(annotation) is list:
        item_annotation = get_args(annotation)[0] if get_args(annotation) else None
        return [
            _normalize_value(item, item_annotation, f"{path}[{index}]", alias_mappings)
            for index, item in enumerate(value)
        ]

    return value


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


def _is_benefit_model(annotation: Any) -> bool:
    from .schema import BenefitTerms

    return _is_model(annotation) and issubclass(annotation, BenefitTerms)


def _is_waiting_status(annotation: Any) -> bool:
    from .schema import WaitingPeriodStatus

    return annotation is WaitingPeriodStatus


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
        return date_parser.parse(cleaned, fuzzy=True).date().isoformat()
    except (ValueError, OverflowError, TypeError) as exc:
        logger.warning("Could not parse date at %s (%r): %s", path, value, exc)
        return None


def _map_waiting_status(value: str) -> str:
    normalized = re.sub(r"[\s_-]+", " ", value.strip().casefold())
    mappings = {
        "covered": "Applied",
        "applicable": "Applied",
        "not applicable": "Not Covered",
        "not covered": "Not Covered",
        "excluded": "Not Covered",
        "waived": "Waived Off",
        "waived off": "Waived Off",
    }
    return mappings.get(normalized, value)


def _normalize_nested_keys(
    field_name: str,
    value: Any,
    alias_mappings: dict[str, dict[str, int]] | None = None,
) -> Any:
    if not isinstance(value, dict):
        return value
    model_type = _nested_model_type(field_name)
    if model_type is None:
        return value
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
