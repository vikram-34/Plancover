from datetime import date
from pathlib import Path

from src.extractor import GMCExtractor
from src.ingest import IngestionResult, PageContent, ingest_pdf
from src.schema import GMCPPolicy


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
                "employee": "Covered; maximum 1 employee",
                "spouse": "Covered",
            },
            "room_rent": "Covered up to 1% of SI, maximum INR 10,000/day",
            "maternity": {
                "waiting_period_status": "Waived Off",
                "baby_day_one_cover": "Covered under Family SI",
            },
            "waiting_periods": {
                "pre_existing_disease": "Applied for 36 months",
            },
            "ambulance": "Covered up to INR 1,000/claim",
        }
    )

    assert policy.policy_period.start_date == date(2024, 4, 1)
    assert policy.ambulance == "Covered up to INR 1,000/claim"


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
        return {}


def test_extractor_handles_document_with_no_matching_info() -> None:
    client = _NoInfoClient()
    document = IngestionResult(
        pages=[PageContent(page_number=1, raw_text="A generic unrelated page.", tables=[])]
    )

    result = GMCExtractor(client=client).extract(document)

    assert client.calls == 6
    assert result.policy is not None
    assert result.policy.insurer_name is None
    assert not result.errors
