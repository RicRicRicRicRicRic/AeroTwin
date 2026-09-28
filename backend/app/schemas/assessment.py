"""Pydantic schemas for building profiles and seismic vulnerability assessment.

Covers Phase 4 of the pipeline: Cross-Frame Result Aggregation → Structural
Assessment. All request bodies and responses for ``backend/app/api/assessment.py``
are defined here.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .processing import JobStatus


class ConstructionType(str, Enum):
    """Supported construction systems (drives the vulnerability lookup table)."""

    RC_FRAME = "rc_frame"
    REINFORCED_MASONRY = "reinforced_masonry"
    UNREINFORCED_MASONRY = "unreinforced_masonry"
    STEEL = "steel"
    TIMBER = "timber"
    ADOBE = "adobe"
    UNKNOWN = "unknown"


class CodeCompliance(str, Enum):
    """Seismic-code compliance indicator for the assessed structure."""

    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    UNKNOWN = "unknown"


class BuildingProfileCreate(BaseModel):
    """Payload for ``POST /api/assessment/profiles``."""

    name: str = Field(min_length=1, max_length=200, description="Unique building name.")
    structure_age_years: float = Field(
        ge=0.0, le=150.0, description="Age of the structure in years at inspection."
    )
    construction_type: ConstructionType
    num_stories: int = Field(default=1, ge=1, le=200)
    code_compliance: CodeCompliance = CodeCompliance.UNKNOWN
    notes: str | None = Field(default=None, max_length=2000)


class BuildingProfileResponse(BuildingProfileCreate):
    """A stored building profile."""

    id: str
    created_at: datetime


class SeismicCalculationRequest(BaseModel):
    """Payload for ``POST /api/assessment/calculate``."""

    profile_id: str = Field(min_length=1, description="Building profile to assess.")
    crack_job_id: str | None = Field(
        default=None,
        description=(
            "Completed crack-mapping job supplying frame defect metrics; "
            "omit for a zero-defect baseline assessment."
        ),
    )
    element_job_id: str | None = Field(
        default=None,
        description=(
            "Completed element-detection job whose masks are compared against the "
            "crack masks to derive element damage states; optional."
        ),
    )


class DefectSummary(BaseModel):
    """Aggregated defect inputs fed into the scoring engine."""

    frames_analyzed: int = 0
    frames_with_cracks: int = 0
    mean_area_ratio: float = 0.0
    mean_width_px: float = 0.0
    mean_length_px: float = 0.0
    longest_crack_px: float = 0.0
    total_crack_pixels: int = 0
    element_damage_states: dict[str, int] | None = Field(
        default=None,
        description="Counts per element damage state (intact/minor/moderate/severe).",
    )


class FactorContribution(BaseModel):
    """One weighted factor of the vulnerability score (audit trail)."""

    name: str
    value: float = Field(ge=0.0, le=1.0, description="Normalised factor value 0-1.")
    weight: float = Field(ge=0.0, le=1.0, description="Normalised weight (sums to 1).")
    contribution: float = Field(ge=0.0, le=1.0, description="weight x value.")


class SeismicAssessmentResult(BaseModel):
    """Deterministic output of the seismic vulnerability scoring engine."""

    vulnerability_score: float = Field(ge=0.0, le=100.0)
    classification: str = Field(description="Low / Moderate / Substantial / Severe / Critical.")
    factors: list[FactorContribution]
    defect_summary: DefectSummary
    recommendations: list[str]


class AssessmentResponse(BaseModel):
    """Status payload for seismic assessments."""

    assessment_id: str
    status: JobStatus
    profile_id: str
    building_name: str
    params: dict[str, Any] = Field(default_factory=dict)
    vulnerability_score: float | None = None
    classification: str | None = None
    result: SeismicAssessmentResult | None = None
    error: str | None = None
    created_at: datetime
    completed_at: datetime | None = None