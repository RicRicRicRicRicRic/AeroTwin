"""Processing API: video ingestion and asynchronous frame extraction jobs.

Endpoints under ``/api/processing``:
* ``POST /ingest-video`` — copy an MP4/MOV into the raw store and register it.
* ``POST /extract-frames`` — queue a background frame-extraction job (HTTP 202).
* ``GET /videos`` / ``GET /jobs`` / ``GET /jobs/{job_id}`` — status queries.

All OpenCV work runs via FastAPI background tasks or the Starlette threadpool,
so the event loop is never blocked (AeroTwin async rule).
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, status
from starlette.concurrency import run_in_threadpool

from ..schemas.processing import (
    ExtractionJobResponse,
    FrameExtractionRequest,
    VideoIngestRequest,
    VideoIngestResponse,
    VideoRecord,
)
from ..services import uav_preprocessor
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


@router.get("/jobs", response_model=list[ExtractionJobResponse], summary="List processing jobs")
async def list_jobs(
    limit: int = Query(default=50, ge=1, le=200),
) -> list[ExtractionJobResponse]:
    """Return the most recent processing jobs, newest first."""
    return await run_in_threadpool(uav_preprocessor.list_extraction_jobs, limit=limit)


@router.get(
    "/jobs/{job_id}",
    response_model=ExtractionJobResponse,
    summary="Get a processing job status",
)
async def get_job(job_id: str) -> ExtractionJobResponse:
    """Return the current status, metrics, and error (if any) of a job."""
    job = await run_in_threadpool(uav_preprocessor.get_extraction_job, job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown processing job: {job_id}",
        )
    return job