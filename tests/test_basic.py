from datetime import date
from pathlib import Path

from src.extractor import GMCExtractor
from src.ingest import IngestionResult, PageContent, ingest_pdf
from src.schema import BenefitStatus, GMCPPolicy, WaitingPeriodStatus


def test_known_good_policy_dict_validates() -> None:
    policy = GMCPPolicy.model_validate(
        {
            "insurer_name": "Example General Insurance Co.",
            "tpa_name": "Example TPA Services",
            "policy_period": {"start_date": "2024-04-01", "end_date": "2025-03-31"},
            "tenure": "1 year",
            "policy_type": "Floater",
            "total_premium": "INR 500,000",
            "total_sum_insured": "INR 10,000,000",
            "family_structure": {
                "employee": {"status": "Covered", "maximum_members": 1},
                "spouse": {"status": "Covered"},
            },
            "room_rent": {
                "status": BenefitStatus.COVERED,
                "percent_of_sum_insured": 1,
                "max_limit": "INR 10,000/day",
            },
            "maternity": {
                "waiting_period_status": WaitingPeriodStatus.WAIVED_OFF,
                "baby_day_one_cover": {"status": "Covered", "limit": "Family SI"},
            },
            "waiting_periods": {
                "pre_existing_disease": {
                    "status": "Applied",
                    "conditions": "36 months",
                }
            },
            "ambulance": {"status": "Covered", "limit": "INR 1,000/claim"},
        }
    )

    assert policy.policy_period.start_date == date(2024, 4, 1)
    assert policy.ambulance.status is BenefitStatus.COVERED


def test_ingest_sample_pdf_returns_text() -> None:
    pdf_path = Path(__file__).parents[1] / "sample_docs" / "GHI_Policy.pdf"
    result = ingest_pdf(pdf_path)

    assert result.page_count > 0
    assert any(page.raw_text.strip() for page in result.pages)


class _NoInfoClient:
    def __init__(self) -> None:
        self.calls = 0

    def complete_json(self, prompt, *, instructions, schema, max_attempts):
        self.calls += 1
        return _not_found_skeleton()


def _not_found_skeleton():
    skeleton = GMCPPolicy().model_dump()

    def mark_missing(value):
        if isinstance(value, dict):
            return {key: mark_missing(item) for key, item in value.items()}
        if isinstance(value, list):
            return value
        return "Not Found"

    return mark_missing(skeleton)


def test_extractor_handles_document_with_no_matching_info() -> None:
    client = _NoInfoClient()
    document = IngestionResult(
        pages=[PageContent(page_number=1, raw_text="A generic unrelated page.", tables=[])]
    )

    result = GMCExtractor(client=client).extract(document)

    assert client.calls == 1
    assert result.policy is not None
    assert result.policy.insurer_name is None
    assert not result.errors
