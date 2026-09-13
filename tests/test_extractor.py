from src.extractor import (
    EXTRACTION_GROUPS,
    build_field_evidence,
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
    assert normalize_not_found({"limit": "Not Found"}) == {"limit": "Not Found"}


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


def test_missing_field_retry_recovers_meaningful_values() -> None:
    class RetryClient:
        def __init__(self) -> None:
            self.calls = 0

        def complete_json(self, prompt, *, instructions, schema, max_attempts):
            self.calls += 1
            if self.calls % 2:
                return {}
            from src.extractor import group_skeleton

            defaults = group_skeleton(tuple(schema["properties"]))
            for field, value in list(defaults.items()):
                if isinstance(value, str):
                    defaults[field] = "Recovered"
            if "policy_period" in defaults:
                defaults["policy_period"] = {"start_date": "2025-01-01"}
            if "total_premium" in defaults:
                defaults["total_premium"] = "Recovered"
            if "total_sum_insured" in defaults:
                defaults["total_sum_insured"] = "Recovered"
            return defaults

    from src.extractor import GMCExtractor

    result = GMCExtractor(client=RetryClient()).extract(
        IngestionResult(pages=[PageContent(1, "Policy period premium room rent", [])])
    )

    assert result.policy is not None
    assert result.policy.total_premium == "Recovered"


def test_field_evidence_contains_page_and_snippet() -> None:
    from src.extractor import GMCExtractor

    policy = GMCExtractor(
        client=type("C", (), {"complete_json": lambda self, prompt, **kwargs: {}})()
    ).extract(
        IngestionResult(pages=[PageContent(4, "Total Premium: INR 500", [])])
    ).policy

    assert policy is not None
    policy = policy.model_copy(update={"total_premium": "INR 500"})
    evidence = build_field_evidence(
        policy, IngestionResult(pages=[PageContent(4, "Total Premium: INR 500", [])])
    )
    assert evidence["total_premium"][0]["page"] == 4
    assert "INR 500" in evidence["total_premium"][0]["snippet"]
