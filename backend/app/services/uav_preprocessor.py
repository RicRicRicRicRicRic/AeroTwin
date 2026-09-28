"""UAV video ingestion and frame extraction service.

Implements Phase 2 of the AeroTwin pipeline:
Video Import → Frame Extraction.

All OpenCV work here is synchronous by design: the API layer runs these
functions in FastAPI background tasks / the threadpool so the event loop is
never blocked. Every filesystem operation uses pathlib, raw videos under
``data/inputs/raw_videos/`` are never overwritten, and extracted frames are
written to a unique run directory under ``data/processed/frames/``.
"""

from __future__ import annotations

import logging
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

from ..core.config import settings
from ..core.database import get_connection
from ..schemas.processing import (
    ExtractionJobResponse,
    ExtractionManifest,
    FrameExtractionRequest,
    FrameExtractionResult,
    JobStatus,
    VideoProbe,
    VideoRecord,
)

logger = logging.getLogger("aerotwin.preprocessing")

#: Only MP4/MOV containers may enter the raw video store.
ALLOWED_VIDEO_EXTENSIONS: frozenset[str] = frozenset({".mp4", ".mov"})

JOB_TYPE_FRAME_EXTRACTION: str = "frame_extraction"


class VideoIngestError(Exception):
    """Base error for video ingestion problems."""


class VideoNotFoundError(VideoIngestError):
    """Raised when a source video or requested raw video does not exist."""


class UnsupportedVideoFormatError(VideoIngestError):
    """Raised when a file extension is not one of the allowed containers."""


class VideoProbeError(VideoIngestError):
    """Raised when OpenCV cannot open a file or its metadata is unusable."""


class FrameExtractionError(Exception):
    """Raised when frame extraction cannot complete (I/O or decode failure)."""


def _utc_now() -> datetime:
    """Timezone-aware current time (all persisted timestamps are UTC ISO-8601)."""
    return datetime.now(timezone.utc)


def probe_video(video_path: Path) -> VideoProbe:
    """Read container metadata (fps, frame count, resolution) via OpenCV.

    Raises:
        VideoNotFoundError: The file does not exist.
        VideoProbeError: OpenCV cannot open it or metadata is unusable.
    """
    if not video_path.is_file():
        raise VideoNotFoundError(f"Video file not found: {video_path}")
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise VideoProbeError(f"OpenCV could not open video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    finally:
        capture.release()
    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        raise VideoProbeError(
            f"Unusable video metadata (fps={fps}, frames={frame_count}, "
            f"{width}x{height}): {video_path}"
        )
    codec = "".join(
        chr((fourcc >> (8 * index)) & 0xFF) for index in range(4)
    ).strip()
    return VideoProbe(
        size_bytes=video_path.stat().st_size,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        duration_seconds=frame_count / fps,
        codec=codec if codec.isprintable() and codec else None,
    )


def _unique_destination(raw_dir: Path, filename: str) -> Path:
    """Pick a destination that never overwrites an existing raw video.

    ``clip.mp4`` → ``clip_1.mp4`` → ``clip_2.mp4`` → ... until free.
    """
    candidate = raw_dir / filename
    if not candidate.exists():
        return candidate
    source = Path(filename)
    counter = 1
    while True:
        candidate = raw_dir / f"{source.stem}_{counter}{source.suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def _build_video_record(video_path: Path, video_id: str | None = None) -> VideoRecord:
    """Probe *video_path* and wrap the metadata into a :class:`VideoRecord`."""
    probe = probe_video(video_path)
    return VideoRecord(
        video_id=video_id if video_id is not None else uuid.uuid4().hex,
        filename=video_path.name,
        stored_path=str(video_path.resolve()),
        size_bytes=probe.size_bytes,
        fps=probe.fps,
        frame_count=probe.frame_count,
        width=probe.width,
        height=probe.height,
        duration_seconds=probe.duration_seconds,
        codec=probe.codec,
        ingested_at=_utc_now(),
    )


def _row_to_video(row: Any) -> VideoRecord:
    """Convert a ``videos`` table row into a :class:`VideoRecord`."""
    return VideoRecord(
        video_id=row["id"],
        filename=row["filename"],
        stored_path=row["stored_path"],
        size_bytes=row["size_bytes"],
        fps=row["fps"],
        frame_count=row["frame_count"],
        width=row["width"],
        height=row["height"],
        duration_seconds=row["duration_seconds"],
        codec=row["codec"],
        ingested_at=datetime.fromisoformat(row["ingested_at"]),
    )


def register_video(record: VideoRecord, db_path: Path | None = None) -> VideoRecord:
    """Persist *record* in the ``videos`` table.

    Idempotent per filename: if the video is already registered (e.g. a file
    dropped into the store manually), the existing row is returned unchanged.
    """
    with get_connection(db_path) as connection:
        existing = connection.execute(
            "SELECT * FROM videos WHERE filename = ?", (record.filename,)
        ).fetchone()
        if existing is not None:
            return _row_to_video(existing)
        connection.execute(
            """
            INSERT INTO videos (
                id, filename, stored_path, size_bytes, fps, frame_count,
                width, height, duration_seconds, codec, ingested_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.video_id,
                record.filename,
                record.stored_path,
                record.size_bytes,
                record.fps,
                record.frame_count,
                record.width,
                record.height,
                record.duration_seconds,
                record.codec,
                record.ingested_at.isoformat(),
            ),
        )
    logger.info("Registered video %s (%s) in aerotwin.db", record.filename, record.video_id)
    return record


def get_video_by_filename(filename: str, db_path: Path | None = None) -> VideoRecord | None:
    """Look up a registered raw video by filename, or ``None``."""
    with get_connection(db_path) as connection:
        row = connection.execute(
            "SELECT * FROM videos WHERE filename = ?", (filename,)
        ).fetchone()
    return _row_to_video(row) if row is not None else None


def list_videos(db_path: Path | None = None) -> list[VideoRecord]:
    """Return every registered raw video, newest ingestion first."""
    with get_connection(db_path) as connection:
        rows = connection.execute(
            "SELECT * FROM videos ORDER BY ingested_at DESC, filename ASC"
        ).fetchall()
    return [_row_to_video(row) for row in rows]


def ingest_video(
    source_path: Path,
    *,
    raw_videos_dir: Path | None = None,
    db_path: Path | None = None,
) -> VideoRecord:
    """Safely copy an MP4/MOV into ``data/inputs/raw_videos/`` and register it.

    Data-integrity guarantees:
    * The source file is only read, never modified.
    * The raw store is never overwritten — colliding names get an incremental
      suffix (``clip.mp4`` → ``clip_1.mp4``).
    * If the source already lives inside the raw store it is registered in
      place without copying.

    Raises:
        VideoNotFoundError: The source does not exist.
        UnsupportedVideoFormatError: Extension outside the allowed set.
        VideoProbeError: OpenCV cannot read the copied file.
    """
    raw_dir = raw_videos_dir if raw_videos_dir is not None else settings.raw_videos_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    source = Path(source_path).expanduser()
    if not source.is_file():
        raise VideoNotFoundError(f"Source video does not exist: {source}")
    if source.suffix.lower() not in ALLOWED_VIDEO_EXTENSIONS:
        raise UnsupportedVideoFormatError(
            f"Unsupported video format {source.suffix!r} for {source.name}; "
            f"allowed: {sorted(ALLOWED_VIDEO_EXTENSIONS)}"
        )
    resolved_source = source.resolve()
    if resolved_source.is_relative_to(raw_dir.resolve()):
        # Already inside the store: register in place, never re-copy.
        record = _build_video_record(resolved_source)
        logger.info("Registered existing raw video %s", record.filename)
        return register_video(record, db_path)
    destination = _unique_destination(raw_dir, source.name)
    shutil.copy2(resolved_source, destination)
    logger.info(
        "Ingested %s -> %s (%d bytes)",
        resolved_source,
        destination.name,
        destination.stat().st_size,
    )
    record = _build_video_record(destination)
    return register_video(record, db_path)


def resolve_raw_video(filename: str, *, raw_videos_dir: Path | None = None) -> Path:
    """Resolve *filename* inside the raw video store, rejecting path traversal.

    Raises:
        VideoNotFoundError: Missing file or a name escaping the store.
        UnsupportedVideoFormatError: Extension outside the allowed set.
    """
    raw_dir = (
        raw_videos_dir if raw_videos_dir is not None else settings.raw_videos_dir
    ).resolve()
    candidate = (raw_dir / filename).resolve()
    if not candidate.is_relative_to(raw_dir):
        raise VideoNotFoundError(f"Filename escapes the raw video directory: {filename!r}")
    if not candidate.is_file():
        raise VideoNotFoundError(f"Video not found in {raw_dir}: {filename}")
    if candidate.suffix.lower() not in ALLOWED_VIDEO_EXTENSIONS:
        raise UnsupportedVideoFormatError(
            f"Unsupported video format {candidate.suffix!r} for {candidate.name}; "
            f"allowed: {sorted(ALLOWED_VIDEO_EXTENSIONS)}"
        )
    return candidate


def ensure_video_record(video_path: Path, db_path: Path | None = None) -> VideoRecord:
    """Return the DB record for a raw video, registering it on the fly if needed."""
    existing = get_video_by_filename(video_path.name, db_path)
    if existing is not None:
        return existing
    logger.warning(
        "Video %s was not ingested through the API; registering it now.",
        video_path.name,
    )
    return register_video(_build_video_record(video_path), db_path)


#: Columns the job-status updater may write (identifiers are never user input).
_JOB_UPDATABLE_COLUMNS: frozenset[str] = frozenset(
    {
        "status",
        "output_dir",
        "frames_written",
        "total_video_frames",
        "processing_time_seconds",
        "error",
        "completed_at",
    }
)


def _row_to_job(row: Any) -> ExtractionJobResponse:
    """Convert a ``processing_jobs`` table row into an API response model."""
    return ExtractionJobResponse(
        job_id=row["id"],
        job_type=row["job_type"],
        status=JobStatus(row["status"]),
        video_filename=row["video_filename"],
        params=FrameExtractionRequest.model_validate_json(row["params_json"]),
        output_dir=row["output_dir"],
        frames_written=row["frames_written"],
        total_video_frames=row["total_video_frames"],
        processing_time_seconds=row["processing_time_seconds"],
        error=row["error"],
        created_at=datetime.fromisoformat(row["created_at"]),
        completed_at=(
            datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None
        ),
    )


def create_extraction_job(
    video: VideoRecord,
    params: FrameExtractionRequest,
    db_path: Path | None = None,
) -> str:
    """Insert a ``pending`` frame-extraction job and return its id."""
    job_id = uuid.uuid4().hex
    with get_connection(db_path) as connection:
        connection.execute(
            """
            INSERT INTO processing_jobs (
                id, job_type, video_id, video_filename, status, params_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                JOB_TYPE_FRAME_EXTRACTION,
                video.video_id,
                video.filename,
                JobStatus.PENDING.value,
                params.model_dump_json(),
                _utc_now().isoformat(),
            ),
        )
    logger.info("Created frame extraction job %s for %s", job_id, video.filename)
    return job_id


def update_extraction_job(
    job_id: str,
    values: dict[str, Any],
    db_path: Path | None = None,
) -> None:
    """Persist selected job columns (``values`` must use whitelisted names).

    Raises:
        KeyError: Unknown job id. ValueError: Non-whitelisted column.
    """
    if not values:
        return
    unknown = set(values) - _JOB_UPDATABLE_COLUMNS
    if unknown:
        raise ValueError(f"Cannot update job columns: {sorted(unknown)}")
    assignments = ", ".join(f"{column} = ?" for column in values)
    sql = f"UPDATE processing_jobs SET {assignments} WHERE id = ?"  # noqa: S608 (whitelisted identifiers)
    with get_connection(db_path) as connection:
        cursor = connection.execute(sql, (*values.values(), job_id))
        if cursor.rowcount == 0:
            raise KeyError(f"Unknown processing job: {job_id}")


def get_extraction_job(job_id: str, db_path: Path | None = None) -> ExtractionJobResponse | None:
    """Fetch a job by id, or ``None`` when it does not exist."""
    with get_connection(db_path) as connection:
        row = connection.execute(
            "SELECT * FROM processing_jobs WHERE id = ?", (job_id,)
        ).fetchone()
    return _row_to_job(row) if row is not None else None


def list_extraction_jobs(
    db_path: Path | None = None,
    limit: int = 50,
) -> list[ExtractionJobResponse]:
    """Return the most recent jobs, newest first."""
    with get_connection(db_path) as connection:
        rows = connection.execute(
            "SELECT * FROM processing_jobs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [_row_to_job(row) for row in rows]


def prepare_extraction_job(
    filename: str,
    params: FrameExtractionRequest,
    *,
    raw_videos_dir: Path | None = None,
    db_path: Path | None = None,
) -> tuple[Path, str]:
    """Validate a raw video, register it if needed, and persist a pending job.

    Returns the resolved video path and the new job id; the caller schedules
    :func:`run_frame_extraction_job` as a background task.
    """
    video_path = resolve_raw_video(filename, raw_videos_dir=raw_videos_dir)
    video = ensure_video_record(video_path, db_path)
    job_id = create_extraction_job(video, params, db_path)
    return video_path, job_id


def extract_frames(
    video_path: Path,
    params: FrameExtractionRequest,
    *,
    frames_dir: Path | None = None,
    job_id: str | None = None,
    video_id: str | None = None,
) -> FrameExtractionResult:
    """Slice *video_path* into numbered frames under ``data/processed/frames/``.

    Frames land in a unique run directory (``<stem>/run_<utc>_<job8>/``), so
    re-runs never overwrite previous outputs, and a ``manifest.json`` records
    the full run configuration for the thesis reproducibility log.

    Sampling:
    * ``frame_step`` — keep frames where ``index % frame_step == 0``.
    * ``seconds_interval`` — keep one frame per interval of video time,
      starting at t=0.

    Raises:
        VideoNotFoundError: The video disappeared.
        VideoProbeError: OpenCV cannot read the container.
        FrameExtractionError: Decode/write failure or zero frames extracted.
    """
    if not video_path.is_file():
        raise VideoNotFoundError(f"Video file not found: {video_path}")
    probe = probe_video(video_path)
    base_dir = frames_dir if frames_dir is not None else settings.frames_dir
    resolved = video_path.resolve()
    run_id = f"run_{_utc_now().strftime('%Y%m%dT%H%M%SZ')}_{(job_id or uuid.uuid4().hex)[:8]}"
    run_dir = base_dir / resolved.stem / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    if params.image_format == "jpg":
        write_params: list[int] = [cv2.IMWRITE_JPEG_QUALITY, params.jpeg_quality]
    else:
        write_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

    started_at = _utc_now()
    capture = cv2.VideoCapture(str(resolved))
    if not capture.isOpened():
        capture.release()
        shutil.rmtree(run_dir, ignore_errors=True)
        raise FrameExtractionError(f"OpenCV could not open video: {resolved}")

    frames_written = 0
    frame_index = 0
    next_seconds = 0.0
    timer_start = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if params.frame_step is not None:
                keep = frame_index % params.frame_step == 0
            else:
                keep = (frame_index / probe.fps) + 1e-9 >= next_seconds
                if keep:
                    next_seconds += params.seconds_interval  # type: ignore[operator]
            if keep:
                frame_path = run_dir / f"frame_{frame_index:06d}.{params.image_format}"
                if not cv2.imwrite(str(frame_path), frame, write_params):
                    raise FrameExtractionError(f"Failed to write frame: {frame_path}")
                frames_written += 1
            frame_index += 1
    except BaseException:
        capture.release()
        # Partial output must never be mistaken for a complete run.
        shutil.rmtree(run_dir, ignore_errors=True)
        raise
    capture.release()
    processing_time_seconds = time.perf_counter() - timer_start

    if frames_written == 0:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise FrameExtractionError(
            f"Decoded 0 frames from {resolved.name} (container reports "
            f"{probe.frame_count} frames)"
        )

    completed_at = _utc_now()
    manifest = ExtractionManifest(
        job_id=job_id or "",
        video_path=str(resolved),
        video_fps=probe.fps,
        video_frame_count=probe.frame_count,
        params=params,
        frames_written=frames_written,
        total_video_frames=probe.frame_count,
        processing_time_seconds=processing_time_seconds,
        started_at=started_at,
        completed_at=completed_at,
        opencv_version=cv2.__version__,
    )
    (run_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )
    logger.info(
        "Extracted %d/%d frames from %s in %.3fs -> %s",
        frames_written,
        probe.frame_count,
        resolved.name,
        processing_time_seconds,
        run_dir,
    )
    return FrameExtractionResult(
        job_id=job_id or "",
        video_id=video_id,
        video_filename=resolved.name,
        output_dir=str(run_dir),
        frames_written=frames_written,
        total_video_frames=probe.frame_count,
        processing_time_seconds=processing_time_seconds,
        started_at=started_at,
        completed_at=completed_at,
    )


def run_frame_extraction_job(
    job_id: str,
    video_path: Path,
    params: FrameExtractionRequest,
    *,
    frames_dir: Path | None = None,
    db_path: Path | None = None,
) -> None:
    """Background-task entry point: run the job and record its outcome.

    Never raises — any failure is captured on the job row
    (``status=failed`` + ``error``) so the API and frontend always have a
    structured message instead of a traceback.
    """
    update_extraction_job(job_id, {"status": JobStatus.RUNNING.value}, db_path)
    try:
        result = extract_frames(video_path, params, frames_dir=frames_dir, job_id=job_id)
    except Exception as exc:  # noqa: BLE001 — job boundary: everything lands on the row
        update_extraction_job(
            job_id,
            {
                "status": JobStatus.FAILED.value,
                "error": str(exc),
                "completed_at": _utc_now().isoformat(),
            },
            db_path,
        )
        logger.exception("Frame extraction job %s failed", job_id)
        return
    update_extraction_job(
        job_id,
        {
            "status": JobStatus.COMPLETED.value,
            "output_dir": result.output_dir,
            "frames_written": result.frames_written,
            "total_video_frames": result.total_video_frames,
            "processing_time_seconds": result.processing_time_seconds,
            "completed_at": result.completed_at.isoformat(),
        },
        db_path,
    )
    logger.info(
        "Frame extraction job %s finished: %d/%d frames in %.3fs",
        job_id,
        result.frames_written,
        result.total_video_frames,
        result.processing_time_seconds,
    )
