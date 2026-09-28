"""Processing API: video ingestion and asynchronous frame extraction jobs.

Endpoints under ``/api/processing``:
* ``POST /ingest-video`` — copy an MP4/MOV into the raw store and register it.
* ``POST /extract-frames`` — queue a background frame-extraction job (HTTP 202).
* ``GET /videos`` / ``GET /jobs`` / ``GET /jobs/{job_id}`` — status queries.

All OpenCV work runs via FastAPI background tasks or the Starlette threadpool,
so the event loop is never blocked (AeroTwin async rule).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable
from urllib.parse import quote

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, status
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from ..core.config import settings
from ..core.database import create_job_record, get_job_record, job_record_to_dict, list_job_records
from ..models import crack_detector, material_segmenter, structural_element_detector
from ..models.base import ModelLoadError, ModelWeightsMissingError
from ..schemas.defect_registry import AggregationRequest, AggregationRun
from ..schemas.processing import (
    ArtifactEntry,
    ArtifactListing,
    CrackMappingRequest,
    ElementDetectionRequest,
    ExtractionJobResponse,
    FrameExtractionRequest,
    FramesJobRequest,
    JobStatus,
    MaterialSegmentationRequest,
    ProcessingJobResponse,
    VideoIngestRequest,
    VideoIngestResponse,
    VideoRecord,
)
from ..services import uav_preprocessor
from ..services.crack_mapper import run_crack_mapping_job
from ..services.cross_frame_aggregator import (
    JOB_TYPE_AGGREGATION,
    load_aggregation_run,
    run_aggregation_job,
)
from ..services.uav_preprocessor import (
    UnsupportedVideoFormatError,
    VideoIngestError,
    VideoNotFoundError,
    VideoProbeError,
)

router = APIRouter(prefix="/api/processing", tags=["processing"])


def _ingest_error(exc: VideoIngestError) -> HTTPException:
    """Map service errors to clean HTTP responses (never a raw traceback)."""
    if isinstance(exc, VideoNotFoundError):
        status_code = status.HTTP_404_NOT_FOUND
    elif isinstance(exc, UnsupportedVideoFormatError):
        status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    elif isinstance(exc, VideoProbeError):
        status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    else:
        status_code = status.HTTP_400_BAD_REQUEST
    return HTTPException(status_code=status_code, detail=str(exc))


@router.post(
    "/ingest-video",
    response_model=VideoIngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Import a raw UAV video",
)
async def ingest_video(request: VideoIngestRequest) -> VideoIngestResponse:
    """Copy an MP4/MOV into data/inputs/raw_videos/ (never overwriting) and register it."""
    try:
        record = await run_in_threadpool(uav_preprocessor.ingest_video, request.source_path)
    except VideoIngestError as exc:
        raise _ingest_error(exc) from exc
    return VideoIngestResponse.model_validate(record.model_dump())


@router.post(
    "/extract-frames",
    response_model=ExtractionJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue a frame extraction job",
)
async def extract_frames(
    request: FrameExtractionRequest,
    background_tasks: BackgroundTasks,
) -> ExtractionJobResponse:
    """Validate the video, persist a pending job, extract frames in the background."""
    try:
        video_path, job_id = await run_in_threadpool(
            uav_preprocessor.prepare_extraction_job, request.filename, request
        )
    except VideoIngestError as exc:
        raise _ingest_error(exc) from exc
    background_tasks.add_task(
        uav_preprocessor.run_frame_extraction_job, job_id, video_path, request
    )
    job = await run_in_threadpool(uav_preprocessor.get_extraction_job, job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Job {job_id} disappeared immediately after creation",
        )
    return job


@router.get("/videos", response_model=list[VideoRecord], summary="List registered raw videos")
async def list_videos() -> list[VideoRecord]:
    """Return every video registered in the raw store, newest first."""
    return await run_in_threadpool(uav_preprocessor.list_videos)


@router.get("/jobs", response_model=list[ProcessingJobResponse], summary="List processing jobs")
async def list_jobs(
    limit: int = Query(default=50, ge=1, le=200),
) -> list[ProcessingJobResponse]:
    """Return the most recent processing jobs of any type, newest first."""
    rows = await run_in_threadpool(list_job_records, limit=limit)
    return [ProcessingJobResponse.model_validate(job_record_to_dict(row)) for row in rows]


@router.get(
    "/jobs/{job_id}",
    response_model=ProcessingJobResponse,
    summary="Get a processing job status",
)
async def get_job(job_id: str) -> ProcessingJobResponse:
    """Return the current status, metrics, and error (if any) of any job."""
    row = await run_in_threadpool(get_job_record, job_id)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown processing job: {job_id}",
        )
    return ProcessingJobResponse.model_validate(job_record_to_dict(row))


# ---------------------------------------------------------------------------
# Phase 3: AI analysis jobs (segmentation, element detection, crack mapping)
# ---------------------------------------------------------------------------
def _resolve_frames_path(frames_path: str) -> Path:
    """Resolve a frames-run reference, mapping failures to clean HTTP errors."""
    try:
        return settings.resolve_frames_run(frames_path)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


def _ensure_weights_available(ensure_fn: Callable[[], Path]) -> None:
    """Run a model's weight check, mapping failures to descriptive HTTP 503."""
    try:
        ensure_fn()
    except (ModelWeightsMissingError, ModelLoadError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc


async def _queue_analysis_job(
    request: FramesJobRequest,
    *,
    job_type: str,
    ensure_weights: Callable[[], Path],
    runner: Callable[..., None],
    background_tasks: BackgroundTasks,
) -> ProcessingJobResponse:
    """Validate inputs/weights, persist a pending job, and schedule its runner.

    Raises HTTP 400/404 for bad frames references and HTTP 503 when model
    weights are missing — before anything is written to the database.
    """
    frames_run_dir = _resolve_frames_path(request.frames_path)
    _ensure_weights_available(ensure_weights)
    job_id = await run_in_threadpool(
        create_job_record,
        job_type=job_type,
        video_filename=frames_run_dir.parent.name,
        params_json=request.model_dump_json(),
    )
    background_tasks.add_task(runner, job_id, frames_run_dir, request)
    row = await run_in_threadpool(get_job_record, job_id)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Job {job_id} disappeared immediately after creation",
        )
    return ProcessingJobResponse.model_validate(job_record_to_dict(row))


@router.post(
    "/segment-materials",
    response_model=ProcessingJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue a material segmentation job",
)
async def segment_materials(
    request: MaterialSegmentationRequest,
    background_tasks: BackgroundTasks,
) -> ProcessingJobResponse:
    """Segment every frame of a run into material class masks (background)."""
    return await _queue_analysis_job(
        request,
        job_type=material_segmenter.JOB_TYPE_MATERIAL_SEGMENTATION,
        ensure_weights=material_segmenter.ensure_model_available,
        runner=material_segmenter.run_material_segmentation_job,
        background_tasks=background_tasks,
    )


@router.post(
    "/detect-elements",
    response_model=ProcessingJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue a structural element detection job",
)
async def detect_elements(
    request: ElementDetectionRequest,
    background_tasks: BackgroundTasks,
) -> ProcessingJobResponse:
    """Detect columns/beams/walls in every frame of a run (background)."""
    return await _queue_analysis_job(
        request,
        job_type=structural_element_detector.JOB_TYPE_ELEMENT_DETECTION,
        ensure_weights=structural_element_detector.ensure_model_available,
        runner=structural_element_detector.run_element_detection_job,
        background_tasks=background_tasks,
    )


@router.post(
    "/map-cracks",
    response_model=ProcessingJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue a crack mapping job",
)
async def map_cracks(
    request: CrackMappingRequest,
    background_tasks: BackgroundTasks,
) -> ProcessingJobResponse:
    """Detect cracks, write crack maps/overlays, and log defect metrics."""
    return await _queue_analysis_job(
        request,
        job_type=crack_detector.JOB_TYPE_CRACK_MAPPING,
        ensure_weights=crack_detector.ensure_model_available,
        runner=run_crack_mapping_job,
        background_tasks=background_tasks,
    )


# ---------------------------------------------------------------------------
# Phase 6: cross-frame aggregation & global defect registry
# ---------------------------------------------------------------------------
async def _require_completed_job(job_id: str, expected_type: str) -> str:
    """Ensure a source job exists, has the right type, and is completed.

    Raises HTTP 404 for an unknown/wrong-type job and HTTP 409 when it has not
    finished yet — both before an aggregation job row is created.
    """
    row = await run_in_threadpool(get_job_record, job_id)
    if row is None or str(row["job_type"]) != expected_type:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No completed {expected_type} job with id {job_id}",
        )
    if str(row["status"]) != JobStatus.COMPLETED.value:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"{expected_type} job {job_id} is {row['status']}; "
                "aggregation requires a completed job"
            ),
        )
    return str(row["video_filename"])


@router.post(
    "/aggregate-results",
    response_model=AggregationRun,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue a cross-frame result aggregation job",
)
async def aggregate_results(
    request: AggregationRequest,
    background_tasks: BackgroundTasks,
) -> AggregationRun:
    """Deduplicate per-frame detections into global defect entities (background)."""
    video_filename = await _require_completed_job(
        request.crack_job_id, crack_detector.JOB_TYPE_CRACK_MAPPING
    )
    if request.element_job_id:
        await _require_completed_job(
            request.element_job_id, structural_element_detector.JOB_TYPE_ELEMENT_DETECTION
        )
    job_id = await run_in_threadpool(
        create_job_record,
        job_type=JOB_TYPE_AGGREGATION,
        video_filename=video_filename,
        params_json=request.model_dump_json(),
    )
    background_tasks.add_task(run_aggregation_job, job_id, request.crack_job_id, request)
    run = await run_in_threadpool(load_aggregation_run, job_id)
    if run is None:  # pragma: no cover — defensive, mirrors the extraction route
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Job {job_id} disappeared immediately after creation",
        )
    return run


@router.get(
    "/aggregations/{job_id}",
    response_model=AggregationRun,
    summary="Get a cross-frame aggregation run and its global defect registry",
)
async def get_aggregation(job_id: str) -> AggregationRun:
    """Return the aggregation status, summary metrics, and global entities.

    Raises HTTP 404 for an unknown id or an id that is not an aggregation run.
    """
    run = await run_in_threadpool(load_aggregation_run, job_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown aggregation job: {job_id}",
        )
    return run


# ---------------------------------------------------------------------------
# Phase 5: artifact browsing for the inspection viewers
# ---------------------------------------------------------------------------
class ArtifactCategory(str, Enum):
    """Managed output directories that the desktop UI may read."""

    FRAMES = "frames"
    MATERIAL_MASKS = "material_masks"
    STRUCTURAL_ELEMENTS = "structural_elements"
    CRACK_MAPS = "crack_maps"
    AGGREGATED = "aggregated"
    REPORTS = "reports"
    ASSESSMENTS = "assessments"


#: File types the artifact API is allowed to stream to the renderer.
ALLOWED_ARTIFACT_SUFFIXES: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".json", ".pdf", ".txt", ".md"}
)
#: Safety cap on directory listings returned to the UI.
MAX_ARTIFACT_ITEMS: int = 500


def _artifact_root(category: ArtifactCategory) -> Path:
    """Map a category onto its managed directory (single source of truth)."""
    roots: dict[ArtifactCategory, Path] = {
        ArtifactCategory.FRAMES: settings.frames_dir,
        ArtifactCategory.MATERIAL_MASKS: settings.material_masks_dir,
        ArtifactCategory.STRUCTURAL_ELEMENTS: settings.structural_elements_dir,
        ArtifactCategory.CRACK_MAPS: settings.crack_maps_dir,
        ArtifactCategory.AGGREGATED: settings.aggregated_dir,
        ArtifactCategory.REPORTS: settings.outputs_dir / "reports",
        ArtifactCategory.ASSESSMENTS: settings.outputs_dir / "assessments",
    }
    return roots[category]


def _artifact_url(category: ArtifactCategory, relative_path: str) -> str:
    """Relative API URL streaming one artifact file."""
    return (
        f"{router.prefix}/artifacts/file"
        f"?category={category.value}&path={quote(relative_path, safe='')}"
    )


@router.get(
    "/artifacts/listing",
    response_model=ArtifactListing,
    summary="List a managed output directory",
)
async def list_artifacts(
    category: ArtifactCategory,
    path: str = Query(default="", description="Path relative to the category root."),
) -> ArtifactListing:
    """Browse frames, masks, crack maps, or report files for the UI viewers."""
    root = _artifact_root(category)
    if not root.is_dir():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Artifact category not available: {category.value}",
        )
    try:
        directory = settings.resolve_managed_path(root, path)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    if not directory.is_dir():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Not a directory: {directory}",
        )

    cleaned = (path or "").strip().replace("\\", "/").strip("/")
    items: list[ArtifactEntry] = []
    for entry in sorted(directory.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))[
        :MAX_ARTIFACT_ITEMS
    ]:
        relative = entry.relative_to(root).as_posix()
        stat = entry.stat()
        items.append(
            ArtifactEntry(
                name=entry.name,
                relative_path=relative,
                url=None if entry.is_dir() else _artifact_url(category, relative),
                is_dir=entry.is_dir(),
                size_bytes=stat.st_size if entry.is_file() else 0,
                modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            )
        )

    parent: str | None = None
    if cleaned:
        parent_path = Path(cleaned).parent.as_posix()
        parent = "" if parent_path == "." else parent_path
    return ArtifactListing(
        category=category.value,
        path=cleaned,
        parent=parent,
        items=items,
    )


@router.get(
    "/artifacts/file",
    summary="Stream one artifact file",
    response_class=FileResponse,
)
async def get_artifact_file(
    category: ArtifactCategory,
    path: str = Query(description="File path relative to the category root."),
) -> FileResponse:
    """Stream a mask/map/report file; traversal-safe and type-restricted."""
    if not path.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="path is required"
        )
    root = _artifact_root(category)
    try:
        target = settings.resolve_managed_path(root, path)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    if not target.is_file():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Not a file: {target}"
        )
    if target.suffix.lower() not in ALLOWED_ARTIFACT_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported artifact type {target.suffix!r}; allowed: "
                f"{sorted(ALLOWED_ARTIFACT_SUFFIXES)}"
            ),
        )
    return FileResponse(str(target), filename=target.name)