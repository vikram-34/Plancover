"""Canonical output schema for extracted Group Medical Cover (GMC) policies.

Insurers describe equivalent cover differently, so the schema separates policy-wide
facts from benefits.  Every coverage-like benefit uses :class:`BenefitTerms`,
giving downstream consumers one predictable ``status``, ``limit``, and ``notes``
shape.  Source-specific wording and exceptions belong in ``notes``; normalised
values belong in the dedicated fields, allowing the extraction layer to remain
faithful to the PDF without making consumers branch on insurer terminology.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class SchemaModel(BaseModel):
    """Base model that rejects unexpected fields in final extraction output."""

    model_config = ConfigDict(extra="forbid")


class BenefitStatus(str, Enum):
    COVERED = "Covered"
    NOT_COVERED = "Not Covered"
    WAIVED_OFF = "Waived Off"


class WaitingPeriodStatus(str, Enum):
    APPLIED = "Applied"
    WAIVED_OFF = "Waived Off"
    NOT_COVERED = "Not Covered"


class BenefitTerms(SchemaModel):
    """Uniform representation for a covered, excluded, or waived benefit."""

    status: Optional[BenefitStatus] = None
    limit: Optional[str] = Field(
        default=None,
        description="Monetary, percentage, visit, or other policy limit as written/normalised.",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Conditions, sub-limits, exclusions, or relevant source wording.",
    )


class PolicyPeriod(SchemaModel):
    start_date: Optional[date] = None
    end_date: Optional[date] = None


class RelationCoverage(BenefitTerms):
    """Benefit terms for a relation, with an optional count limit."""

    maximum_members: Optional[int] = Field(default=None, ge=0)


class FamilyStructure(SchemaModel):
    employee: RelationCoverage = Field(default_factory=RelationCoverage)
    spouse: RelationCoverage = Field(default_factory=RelationCoverage)
    children: RelationCoverage = Field(default_factory=RelationCoverage)
    parents: RelationCoverage = Field(default_factory=RelationCoverage)
    parents_in_law: RelationCoverage = Field(default_factory=RelationCoverage)


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


class AccommodationCharges(BenefitTerms):
    """Coverage terms for room rent or ICU charges."""

    percent_of_sum_insured: Optional[float] = Field(default=None, ge=0)
    max_limit: Optional[str] = Field(default=None, description="Absolute cap, such as INR 5,000/day.")


class HospitalizationPeriods(SchemaModel):
    pre_hospitalization_days: Optional[int] = Field(default=None, ge=0)
    post_hospitalization_days: Optional[int] = Field(default=None, ge=0)
    notes: Optional[str] = None


class LocationLimits(SchemaModel):
    metro: Optional[str] = None
    non_metro: Optional[str] = None
    notes: Optional[str] = None


class MaternityCoverage(SchemaModel):
    waiting_period_status: Optional[WaitingPeriodStatus] = None
    waiting_period_conditions: Optional[str] = None
    baby_day_one_cover: BenefitTerms = Field(default_factory=BenefitTerms)
    vaccination_coverage: BenefitTerms = Field(default_factory=BenefitTerms)
    normal_delivery_limits: LocationLimits = Field(default_factory=LocationLimits)
    c_section_limits: LocationLimits = Field(default_factory=LocationLimits)
    notes: Optional[str] = None


class WaitingPeriod(BenefitTerms):
    """Waiting-period terms, with status and conditions retained for auditability."""

    status: Optional[WaitingPeriodStatus] = None
    conditions: Optional[str] = None


class WaitingPeriods(SchemaModel):
    initial_30_day: WaitingPeriod = Field(default_factory=WaitingPeriod)
    first_year: WaitingPeriod = Field(default_factory=WaitingPeriod)
    second_year: WaitingPeriod = Field(default_factory=WaitingPeriod)
    pre_existing_disease: WaitingPeriod = Field(default_factory=WaitingPeriod)


class OtherBenefits(SchemaModel):
    day_care: BenefitTerms = Field(default_factory=BenefitTerms)
    opd: BenefitTerms = Field(default_factory=BenefitTerms)
    teleconsultation: BenefitTerms = Field(default_factory=BenefitTerms)
    pharmacy_discount: BenefitTerms = Field(default_factory=BenefitTerms)
    domiciliary_hospitalization: BenefitTerms = Field(default_factory=BenefitTerms)
    annual_health_checkup: BenefitTerms = Field(default_factory=BenefitTerms)
    modern_treatment: BenefitTerms = Field(default_factory=BenefitTerms)
    bariatric_treatment: BenefitTerms = Field(default_factory=BenefitTerms)
    psychiatric_treatment: BenefitTerms = Field(default_factory=BenefitTerms)
    ayush_treatment: BenefitTerms = Field(default_factory=BenefitTerms)
    lgbtq_plus_coverage: BenefitTerms = Field(default_factory=BenefitTerms)
    live_in_partner_coverage: BenefitTerms = Field(default_factory=BenefitTerms)
    organ_donor_expenses: BenefitTerms = Field(default_factory=BenefitTerms)


class GMCPPolicy(SchemaModel):
    """Complete normalised extraction record for one GMC policy PDF."""

    insurer_name: Optional[str] = None
    tpa_name: Optional[str] = None
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

    room_rent: AccommodationCharges = Field(default_factory=AccommodationCharges)
    icu_charges: AccommodationCharges = Field(default_factory=AccommodationCharges)
    hospitalization_periods: HospitalizationPeriods = Field(default_factory=HospitalizationPeriods)
    maternity: MaternityCoverage = Field(default_factory=MaternityCoverage)
    waiting_periods: WaitingPeriods = Field(default_factory=WaitingPeriods)
    other_benefits: OtherBenefits = Field(default_factory=OtherBenefits)

    infertility_surrogacy: BenefitTerms = Field(default_factory=BenefitTerms)
    ambulance: BenefitTerms = Field(default_factory=BenefitTerms)
    air_ambulance: BenefitTerms = Field(default_factory=BenefitTerms)
    corporate_buffer_limit: BenefitTerms = Field(default_factory=BenefitTerms)
    disease_wise_capping: BenefitTerms = Field(default_factory=BenefitTerms)
