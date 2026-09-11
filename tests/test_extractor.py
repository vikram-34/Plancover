from src.extractor import (
    EXTRACTION_GROUPS,
    chunk_document,
    group_json_schema,
    normalize_group_keys,
    normalize_not_found,
    select_relevant_chunks,
)
from src.ingest import IngestionResult, PageContent
from src.llm_client import _parse_json_object


def test_keyword_routing_and_not_found_normalisation() -> None:
    document = IngestionResult(
        pages=[
            PageContent(page_number=1, raw_text="Policy details and sum insured", tables=[]),
            PageContent(page_number=2, raw_text="Room Rent is 1% of sum insured", tables=[]),
            PageContent(page_number=3, raw_text="Maternity normal delivery cover", tables=[]),
        ]
    )
    chunks = chunk_document(document)
    room_group = next(group for group in EXTRACTION_GROUPS if group.name == "room_rent_and_icu")

    selected = select_relevant_chunks(room_group, chunks, document.pages)

    assert any(2 in chunk.page_numbers for chunk in selected)
    assert normalize_not_found({"limit": "Not Found"}) == {"limit": None}


def test_group_schema_exposes_only_group_fields() -> None:
    room_group = next(group for group in EXTRACTION_GROUPS if group.name == "room_rent_and_icu")

    schema = group_json_schema(room_group)

    assert set(schema["properties"]) == {"room_rent", "icu_charges"}


def test_fenced_json_is_repaired_before_parsing() -> None:
    assert _parse_json_object('```json\n{"status": "Covered"}\n```') == {"status": "Covered"}


def test_near_miss_and_flattened_keys_are_normalized() -> None:
    normalized = normalize_group_keys(
        {
            "Room Rent": {"Max Limit": "INR 5,000/day"},
            "Policy Period - Start Date": "2025-04-01",
            "unrelated field": "ignored later",
        },
        ("room_rent", "policy_period"),
    )

    assert normalized["room_rent"]["max_limit"] == "INR 5,000/day"
    assert normalized["policy_period"]["start_date"] == "2025-04-01"
