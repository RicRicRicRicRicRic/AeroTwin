"""Pydantic schemas for structural inspection report generation (Phase 4).

Covers the final pipeline stage: Report Generation. Used by
``backend/app/api/reports.py`` and ``backend/app/services/report_generator.py``.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .assessment import BuildingProfileResponse, SeismicAssessmentResult
from .processing import JobStatus


class ReportFormat(str, Enum):
    """Output formats for generated inspection reports."""

    JSON = "json"
    PDF = "pdf"
    BOTH = "both"


class ReportGenerateRequest(BaseModel):
    """Payload for ``POST /api/reports/generate``."""

    assessment_id: str = Field(min_length=1, description="Completed assessment to report.")
    report_format: ReportFormat = ReportFormat.BOTH
    include_visual_maps: bool = Field(
        default=True,
        description="Embed crack overlay images (when available) in the PDF report.",
    )


class ReportResponse(BaseModel):
    """Status payload for report generation records."""

    report_id: str
    assessment_id: str
    status: JobStatus
    report_format: ReportFormat
    json_path: str | None = None
    pdf_path: str | None = None
    assessment_json_path: str | None = None
    size_bytes: int | None = None
    error: str | None = None
    created_at: datetime
    completed_at: datetime | None = None


class ReportSummary(BaseModel):
    """Structured JSON summary written to ``data/outputs/reports/``."""

    report_id: str
    assessment_id: str
    generated_at: datetime
    application: str
    application_version: str
    building: BuildingProfileResponse
    result: SeismicAssessmentResult
    visual_maps: list[str] = Field(
        default_factory=list,
        description="Absolute paths of defect overlay images referenced by the report.",
    )
    pipeline: dict[str, Any] = Field(
        default_factory=dict,
        description="Source job ids/output dirs behind this report (provenance).",
    )