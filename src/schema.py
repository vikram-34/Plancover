"""Flat canonical output schema for extracted GMC policies.

The schema deliberately uses descriptive strings for coverage terms instead of
deep ``status``/``limit``/``notes`` objects.  This keeps the extraction target
reliable for smaller local models while preserving the original policy wording
in a single auditable value.  Only information that benefits from structure,
such as policy dates, demographics, and sum-insured tiers, remains nested.
"""

from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


NOT_FOUND = "Not Found"


class SchemaModel(BaseModel):
    """Base model that rejects unexpected fields in final extraction output."""

    model_config = ConfigDict(extra="forbid")


class PolicyPeriod(SchemaModel):
    start_date: Optional[date] = None
    end_date: Optional[date] = None


class FamilyStructure(SchemaModel):
    employee: str = NOT_FOUND
    spouse: str = NOT_FOUND
    children: str = NOT_FOUND
    parents: str = NOT_FOUND
    parents_in_law: str = NOT_FOUND


class SumInsuredTier(SchemaModel):
    name: Optional[str] = Field(default=None, description="Tier label, for example 'Employee + Family'.")
    amount: Optional[str] = Field(default=None, description="Sum insured including currency, if stated.")
    eligible_members: Optional[str] = None
    notes: Optional[str] = None


class DemographicCounts(SchemaModel):
    employees: Optional[int] = Field(default=None, ge=0)
    spouses: Optional[int] = Field(default=None, ge=0)
    children: Optional[int] = Field(default=None, ge=0)
    parents: Optional[int] = Field(default=None, ge=0)
    parents_in_law: Optional[int] = Field(default=None, ge=0)
    male: Optional[int] = Field(default=None, ge=0)
    female: Optional[int] = Field(default=None, ge=0)
    other: Optional[int] = Field(default=None, ge=0)
    notes: Optional[str] = None


class HospitalizationPeriods(SchemaModel):
    pre_hospitalization_days: Optional[int] = Field(default=None, ge=0)
    post_hospitalization_days: Optional[int] = Field(default=None, ge=0)
    notes: Optional[str] = None


class EvidenceItem(SchemaModel):
    page: int
    snippet: str


class MaternityCoverage(SchemaModel):
    waiting_period_status: str = NOT_FOUND
    waiting_period_conditions: str = NOT_FOUND
    baby_day_one_cover: str = NOT_FOUND
    vaccination_coverage: str = NOT_FOUND
    normal_delivery_limits: str = NOT_FOUND
    c_section_limits: str = NOT_FOUND
    notes: str = NOT_FOUND


class WaitingPeriods(SchemaModel):
    initial_30_day: str = NOT_FOUND
    first_year: str = NOT_FOUND
    second_year: str = NOT_FOUND
    pre_existing_disease: str = NOT_FOUND


class OtherBenefits(SchemaModel):
    day_care: str = NOT_FOUND
    opd: str = NOT_FOUND
    teleconsultation: str = NOT_FOUND
    pharmacy_discount: str = NOT_FOUND
    domiciliary_hospitalization: str = NOT_FOUND
    annual_health_checkup: str = NOT_FOUND
    modern_treatment: str = NOT_FOUND
    bariatric_treatment: str = NOT_FOUND
    psychiatric_treatment: str = NOT_FOUND
    ayush_treatment: str = NOT_FOUND
    lgbtq_plus_coverage: str = NOT_FOUND
    live_in_partner_coverage: str = NOT_FOUND
    organ_donor_expenses: str = NOT_FOUND


class GMCPPolicy(SchemaModel):
    """Complete, model-friendly normalised extraction record for one policy PDF."""

    insurer_name: Optional[str] = None
    tpa_name: Optional[str] = None
    extraction_status: Optional[Literal["succeeded", "low_quality"]] = Field(
        default=None,
        description="Pipeline quality gate result; not a policy term.",
    )
    extraction_evidence: dict[str, list[EvidenceItem]] = Field(default_factory=dict)
    policy_period: PolicyPeriod = Field(default_factory=PolicyPeriod)
    tenure: Optional[str] = Field(default=None, description="Policy tenure as stated, e.g. '1 year'.")
    policy_type: Optional[str] = None
    total_premium: Optional[str] = None
    total_sum_insured: Optional[str] = None
    previous_year_premium: Optional[str] = None

    family_structure: FamilyStructure = Field(default_factory=FamilyStructure)
    sum_insured_tiers: list[SumInsuredTier] = Field(default_factory=list)
    demographics_counts: DemographicCounts = Field(default_factory=DemographicCounts)
    total_lives_covered: Optional[int] = Field(default=None, ge=0)

    room_rent: str = NOT_FOUND
    icu_charges: str = NOT_FOUND
    hospitalization_periods: HospitalizationPeriods = Field(default_factory=HospitalizationPeriods)
    maternity: MaternityCoverage = Field(default_factory=MaternityCoverage)
    waiting_periods: WaitingPeriods = Field(default_factory=WaitingPeriods)
    other_benefits: OtherBenefits = Field(default_factory=OtherBenefits)

    infertility_surrogacy: str = NOT_FOUND
    ambulance: str = NOT_FOUND
    air_ambulance: str = NOT_FOUND
    corporate_buffer_limit: str = NOT_FOUND
    disease_wise_capping: str = NOT_FOUND
