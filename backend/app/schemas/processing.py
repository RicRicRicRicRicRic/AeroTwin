"""Pydantic schemas for UAV video ingestion, job control, and frame extraction.

Covers Phase 2 of the pipeline: Video Import → Frame Extraction. All request
bodies and responses used by ``backend/app/api/processing.py`` live here.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class JobStatus(str, Enum):
    """Lifecycle states of a processing job recorded in ``aerotwin.db``."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class VideoIngestRequest(BaseModel):
    """Payload for ``POST /api/processing/ingest-video``."""

    source_path: Path = Field(
        description=(
            "Absolute path of the MP4/MOV file to copy into "
            "data/inputs/raw_videos/ (existing files are never overwritten)."
        )
    )


class VideoProbe(BaseModel):
    """Container metadata read from a video file via OpenCV."""

    size_bytes: int = Field(description="File size in bytes.")
    fps: float = Field(description="Frames per second reported by the container.")
    frame_count: int = Field(description="Total frames reported by the container.")
    width: int = Field(description="Frame width in pixels.")
    height: int = Field(description="Frame height in pixels.")
    duration_seconds: float = Field(description="frame_count / fps.")
    codec: str | None = Field(default=None, description="FourCC codec tag when available.")


class VideoRecord(BaseModel):
    """A raw UAV video registered in ``aerotwin.db``."""

    video_id: str
    filename: str = Field(description="Filename inside data/inputs/raw_videos/.")
    stored_path: str = Field(description="Absolute path of the stored raw video.")
    size_bytes: int
    fps: float
    frame_count: int
    width: int
    height: int
    duration_seconds: float
    codec: str | None = None
    ingested_at: datetime


class VideoIngestResponse(VideoRecord):
    """Response returned with HTTP 201 after a successful ingestion."""


class FrameExtractionRequest(BaseModel):
    """Frame sampling parameters; exactly one sampling mode must be chosen.

    * ``frame_step`` — keep every Nth frame (e.g. 30 → one frame per second
      of a 30 fps video).
    * ``seconds_interval`` — keep one frame every N seconds of video time
      (robust to variable frame rates).
    """

    filename: str = Field(description="Video filename inside data/inputs/raw_videos/.")
    frame_step: int | None = Field(
        default=None,
        ge=1,
        description="Keep every Nth frame. Mutually exclusive with seconds_interval.",
    )
    seconds_interval: float | None = Field(
        default=None,
        gt=0,
        description="Keep one frame every N seconds. Mutually exclusive with frame_step.",
    )
    image_format: Literal["jpg", "png"] = Field(
        default="jpg",
        description="Output format for extracted frames.",
    )
    jpeg_quality: int = Field(
        default=95,
        ge=1,
        le=100,
        description="JPEG quality 1-100 (ignored when image_format is png).",
    )

    @model_validator(mode="after")
    def _validate_sampling_mode(self) -> FrameExtractionRequest:
        """Enforce XOR between the two sampling modes for reproducible logs."""
        if (self.frame_step is None) == (self.seconds_interval is None):
            raise ValueError("Provide exactly one of 'frame_step' or 'seconds_interval'.")
        return self


class FrameExtractionResult(BaseModel):
    """Service-level outcome of a completed frame extraction run."""

    job_id: str
    video_id: str | None = None
    video_filename: str
    output_dir: str = Field(description="Unique run directory holding the frames.")
    frames_written: int
    total_video_frames: int = Field(description="Frames reported by the video container.")
    processing_time_seconds: float = Field(description="Wall-clock time of the extraction loop.")
    started_at: datetime
    completed_at: datetime


class ExtractionJobResponse(BaseModel):
    """Status payload for processing jobs (``GET /api/processing/jobs/{job_id}``)."""

    job_id: str
    job_type: str = "frame_extraction"
    status: JobStatus
    video_filename: str
    params: FrameExtractionRequest = Field(description="Sampling parameters used for the job.")
    output_dir: str | None = None
    frames_written: int | None = None
    total_video_frames: int | None = None
    processing_time_seconds: float | None = None
    error: str | None = Field(default=None, description="Failure detail when status=failed.")
    created_at: datetime
    completed_at: datetime | None = None


class ExtractionManifest(BaseModel):
    """Sidecar manifest written next to extracted frames for reproducibility."""

    job_id: str
    video_path: str
    video_fps: float
    video_frame_count: int
    params: FrameExtractionRequest
    frames_written: int
    total_video_frames: int
    processing_time_seconds: float
    started_at: datetime
    completed_at: datetime
    opencv_version: str


# ---------------------------------------------------------------------------
# Phase 3: AI analysis jobs (segmentation, element detection, crack mapping)
# ---------------------------------------------------------------------------
class FramesJobRequest(BaseModel):
    """Shared request for AI analysis jobs operating on an extracted-frames run."""

    frames_path: str = Field(
        min_length=1,
        description=(
            "Frames-run reference relative to data/processed/frames/, "
            "e.g. 'smoke_test/run_20260928T075843Z_7f201442'."
        ),
    )


class MaterialSegmentationRequest(FramesJobRequest):
    """Payload for ``POST /api/processing/segment-materials``."""

    tile_size: int | None = Field(
        default=None,
        ge=64,
        le=2048,
        description="Optional inference tile-size override (memory/quality trade-off).",
    )


class ElementDetectionRequest(FramesJobRequest):
    """Payload for ``POST /api/processing/detect-elements``."""


class CrackMappingRequest(FramesJobRequest):
    """Payload for ``POST /api/processing/map-cracks``."""

    crack_threshold: float = Field(
        default=0.5,
        gt=0.0,
        lt=1.0,
        description="Crack-probability threshold for binarising masks (0-1).",
    )


class ProcessingJobResponse(BaseModel):
    """Generic status payload for any processing job type.

    Used by the shared ``GET /api/processing/jobs`` endpoints; ``params`` and
    ``metrics`` stay untyped JSON so extraction, segmentation, and
    crack-mapping jobs serialise through one model.
    """

    job_id: str
    job_type: str
    status: JobStatus
    video_filename: str
    params: dict[str, Any] = Field(default_factory=dict)
    output_dir: str | None = None
    frames_written: int | None = None
    total_video_frames: int | None = None
    processing_time_seconds: float | None = None
    metrics: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime
    completed_at: datetime | None = None


class DetectedElement(BaseModel):
    """A bounding box for one detected structural element."""

    label: str = Field(description="One of ELEMENT_CLASSES (column/beam/wall).")
    x: int
    y: int
    width: int
    height: int
    confidence: float = Field(ge=0.0, le=1.0)


class CrackFrameMetrics(BaseModel):
    """Per-frame crack defect metrics (persisted to ``frame_defect_metrics``)."""

    frame_filename: str
    crack_pixel_count: int = Field(ge=0)
    crack_area_ratio: float = Field(ge=0.0, description="Crack pixels / total pixels x 100.")
    crack_length_px: float = Field(ge=0.0, description="Skeleton length in px (Zhang-Suen).")
    mean_width_px: float = Field(ge=0.0, description="Crack pixels / skeleton length.")
    component_count: int = Field(ge=0)


class CrackRunSummary(BaseModel):
    """Run-level aggregate of per-frame crack metrics (stored in ``metrics_json``)."""

    frames_analyzed: int
    frames_with_cracks: int
    total_crack_pixels: int
    mean_area_ratio: float
    mean_crack_length_px: float
    mean_width_px: float
    longest_crack_px: float


class AnalysisRunManifest(BaseModel):
    """Sidecar manifest written next to AI analysis outputs (reproducibility)."""

    job_id: str
    job_type: str
    frames_run: str
    frames_run_path: str
    frames_processed: int
    params: dict[str, Any]
    class_names: list[str] | None = None
    weights_file: str
    device: str
    tile_size: int | None = None
    torch_version: str
    opencv_version: str
    processing_time_seconds: float
    started_at: datetime
    completed_at: datetime


# ---------------------------------------------------------------------------
# Phase 5: artifact browsing for the inspection viewers
# ---------------------------------------------------------------------------
class ArtifactEntry(BaseModel):
    """One file or directory inside a managed output directory."""

    name: str
    relative_path: str = Field(description="Path relative to the category root.")
    url: str | None = Field(
        default=None,
        description="Relative API URL streaming the file (None for directories).",
    )
    is_dir: bool
    size_bytes: int
    modified_at: datetime


class ArtifactListing(BaseModel):
    """Directory listing for a managed output category (frontend browser)."""

    category: str
    path: str
    parent: str | None = Field(
        default=None, description="Parent path relative to the category root, if any."
    )
    items: list[ArtifactEntry]

