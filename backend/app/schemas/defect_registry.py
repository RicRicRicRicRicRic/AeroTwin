"""Pydantic schemas for the global defect registry (Phase 6: aggregation).

Phase 6 converts *frame-local* detections ("5 crack components in frame 42")
into *global* defect entities ("Crack CRK-0001 observed in frames 40-55, max
length 124 px") by spatially deduplicating overlapping observations across
adjacent frames. Every model in this module is part of either

* the aggregation job payload/response (``AggregationRequest`` /
  ``AggregationRun``),
* the persisted registry (``GlobalCrackEntity`` / ``GlobalElementEntity`` and
  their ``DefectObservation`` provenance records), or
* the sidecar artifact ``aggregated_defects.json`` (``DefectRegistry``).

All geometry is expressed in frame pixel coordinates as axis-aligned bounding
boxes (``x``/``y``/``width``/``height``, the same flat style as
:class:`~app.schemas.processing.DetectedElement`).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from .processing import JobStatus


class DefectObservation(BaseModel):
    """One frame-local detection merged into a global entity."""

    frame_filename: str = Field(description="Frame image filename the observation came from.")
    frame_index: int = Field(ge=0, description="Numeric frame index parsed from the filename.")
    x: int = Field(ge=0, description="Bounding-box left edge in frame pixels.")
    y: int = Field(ge=0, description="Bounding-box top edge in frame pixels.")
    width: int = Field(ge=1, description="Bounding-box width in frame pixels.")
    height: int = Field(ge=1, description="Bounding-box height in frame pixels.")
    pixel_count: int = Field(default=0, ge=0, description="Mask pixels of the component.")
    length_px: float = Field(
        default=0.0, ge=0.0, description="Skeleton length of the component in px (cracks only)."
    )
    width_px: float = Field(
        default=0.0, ge=0.0, description="Mean stroke width (pixels / length) in px."
    )
    confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Detector confidence (elements only)."
    )
    label: str | None = Field(
        default=None,
        description="Element class of this observation (column/beam/wall); None for cracks.",
    )
    crack_density: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Crack-pixel density inside this element observation (0-1); only "
            "computed when both an element and a crack job were supplied."
        ),
    )
    match_iou: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Bounding-box IoU against the previous observation of the same entity "
            "at merge time (None for the observation that created the entity)."
        ),
    )


class GlobalCrackEntity(BaseModel):
    """One physical crack, deduplicated across every frame it appears in."""

    defect_id: str = Field(description="Stable registry id, e.g. CRK-0001.")
    kind: Literal["crack"] = "crack"
    label: str = Field(default="crack", description="Defect class label (crack).")
    observation_count: int = Field(ge=1)
    first_frame_filename: str
    last_frame_filename: str
    first_frame_index: int = Field(ge=0)
    last_frame_index: int = Field(ge=0)
    union_x: int = Field(ge=0, description="Union bounding box of all observations.")
    union_y: int = Field(ge=0)
    union_width: int = Field(ge=1)
    union_height: int = Field(ge=1)
    pixel_count_max: float = Field(ge=0.0, description="Largest single-frame crack area.")
    pixel_count_mean: float = Field(ge=0.0, description="Mean single-frame crack area.")
    length_px_max: float = Field(ge=0.0, description="Longest observed skeleton length.")
    length_px_mean: float = Field(ge=0.0)
    width_px_max: float = Field(ge=0.0, description="Widest observed mean stroke width.")
    width_px_mean: float = Field(ge=0.0)
    mean_match_iou: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Mean merge IoU (None when single-observation)."
    )
    representative: DefectObservation = Field(
        description=(
            "Strongest observation of the entity (longest crack / largest area); the "
            "single-frame record quoted as this defect's severity evidence."
        )
    )
    observations: list[DefectObservation] = Field(
        default_factory=list,
        description="Per-frame provenance; empty unless include_observations was requested.",
    )


class GlobalElementEntity(BaseModel):
    """One physical structural element, deduplicated across frames."""

    element_id: str = Field(description="Stable registry id, e.g. ELM-0001.")
    kind: Literal["element"] = "element"
    label: str = Field(description="Element class (column/beam/wall).")
    observation_count: int = Field(ge=1)
    first_frame_filename: str
    last_frame_filename: str
    first_frame_index: int = Field(ge=0)
    last_frame_index: int = Field(ge=0)
    union_x: int = Field(ge=0)
    union_y: int = Field(ge=0)
    union_width: int = Field(ge=1)
    union_height: int = Field(ge=1)
    confidence_max: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_mean: float | None = Field(default=None, ge=0.0, le=1.0)
    crack_density_max: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Worst crack density observed inside this element; the seismic engine "
            "maps it onto the element damage state (None without crack masks)."
        ),
    )
    mean_match_iou: float | None = Field(default=None, ge=0.0, le=1.0)
    representative: DefectObservation
    observations: list[DefectObservation] = Field(default_factory=list)


class AggregationSummary(BaseModel):
    """Run-level effect of deduplication (stored in ``metrics_json``)."""

    frames_analyzed: int = Field(ge=0, description="Frames carrying any input observation.")
    frames_with_cracks: int = Field(ge=0)
    raw_crack_observations: int = Field(ge=0, description="Frame-level crack components read.")
    unique_crack_defects: int = Field(ge=0, description="Global crack entities after dedup.")
    raw_element_observations: int = Field(default=0, ge=0)
    unique_element_instances: int | None = Field(
        default=None, description="None when no element-detection job was supplied."
    )
    mean_observations_per_crack: float = Field(ge=0.0)
    crack_dedup_ratio: float = Field(
        ge=0.0, le=1.0, description="1 - unique/raw crack observations (0 = no duplicates)."
    )
    crack_dedup_reduction: int = Field(ge=0, description="Duplicate observations folded away.")
    elements_by_label: dict[str, int] | None = Field(
        default=None, description="Unique element count per class label."
    )
    iou_threshold: float = Field(ge=0.0, le=1.0)
    max_frame_gap: int = Field(ge=0, description="Max frame gap still treated as the same entity.")
    method: str = Field(default="greedy-bbox-iou")


class AggregationRequest(BaseModel):
    """Payload for ``POST /api/processing/aggregate-results``."""

    crack_job_id: str = Field(min_length=1, description="Completed crack-mapping job.")
    element_job_id: str | None = Field(
        default=None,
        description="Completed element-detection job to deduplicate alongside the cracks.",
    )
    iou_threshold: float = Field(
        default=0.3,
        gt=0.0,
        le=1.0,
        description="Minimum bounding-box IoU for two observations to be the same defect.",
    )
    max_frame_gap: int = Field(
        default=10,
        ge=0,
        le=100000,
        description=(
            "Maximum frame distance within which observations may be merged. UAV "
            "sequences move continuously, so a larger gap means a different defect."
        ),
    )
    include_observations: bool = Field(
        default=True,
        description="Persist full per-frame provenance (recommended for the thesis audit trail).",
    )


class DefectRegistry(BaseModel):
    """``aggregated_defects.json`` sidecar: the whole global registry + provenance."""

    version: str = Field(default="1.0")
    aggregation_job_id: str
    crack_job_id: str
    element_job_id: str | None = None
    video_filename: str
    generated_at: datetime
    params: dict[str, Any] = Field(default_factory=dict)
    summary: AggregationSummary
    cracks: list[GlobalCrackEntity] = Field(default_factory=list)
    elements: list[GlobalElementEntity] = Field(default_factory=list)


class AggregationRun(BaseModel):
    """Response of ``GET /api/processing/aggregations/{job_id}``."""

    job_id: str
    status: JobStatus
    video_filename: str
    crack_job_id: str | None = None
    element_job_id: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    summary: AggregationSummary | None = None
    cracks: list[GlobalCrackEntity] = Field(default_factory=list)
    elements: list[GlobalElementEntity] = Field(default_factory=list)
    output_dir: str | None = None
    error: str | None = None
    processing_time_seconds: float | None = None
    created_at: datetime
    completed_at: datetime | None = None

