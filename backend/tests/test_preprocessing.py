"""Unit and API tests for Phase 2: UAV video ingestion & frame extraction.

Service tests use explicit tmp_path directories; API tests rely on
``conftest.py`` which redirects ``AEROTWIN_DATA_DIR`` to a throw-away folder
so the real ``data/aerotwin.db`` is never touched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.database import init_database
from app.main import app
from app.schemas.processing import ExtractionManifest, FrameExtractionRequest, JobStatus
from app.services.uav_preprocessor import (
    UnsupportedVideoFormatError,
    VideoNotFoundError,
    extract_frames,
    get_extraction_job,
    get_video_by_filename,
    ingest_video,
    prepare_extraction_job,
    resolve_raw_video,
    run_frame_extraction_job,
)

FRAME_COUNT = 30
VIDEO_FPS = 10.0
VIDEO_SIZE: tuple[int, int] = (64, 48)


def write_synthetic_video(path: Path) -> Path:
    """Create a small deterministic MP4 (30 frames @ 10 fps, 64x48)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, VIDEO_SIZE
    )
    assert writer.isOpened(), "OpenCV VideoWriter could not open (mp4v unavailable)"
    try:
        for index in range(FRAME_COUNT):
            frame = np.full(
                (VIDEO_SIZE[1], VIDEO_SIZE[0], 3), (index * 8) % 255, dtype=np.uint8
            )
            writer.write(frame)
    finally:
        writer.release()
    assert path.is_file() and path.stat().st_size > 0, "synthetic video was not written"
    return path


@pytest.fixture
def raw_videos_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "raw_videos"
    directory.mkdir()
    return directory


@pytest.fixture
def frames_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "frames"
    directory.mkdir()
    return directory


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return init_database(tmp_path / "aerotwin.db")


@pytest.fixture
def source_video(tmp_path: Path) -> Path:
    return write_synthetic_video(tmp_path / "uploads" / "survey.mp4")


# ---------------------------------------------------------------------------
# Ingestion (Video Import)
# ---------------------------------------------------------------------------
def test_ingest_copies_video_and_registers_it(
    source_video: Path, raw_videos_dir: Path, db_path: Path
) -> None:
    original_bytes = source_video.read_bytes()

    record = ingest_video(source_video, raw_videos_dir=raw_videos_dir, db_path=db_path)

    stored = raw_videos_dir / "survey.mp4"
    assert stored.is_file()
    # The source file is read-only input: contents must be untouched.
    assert source_video.read_bytes() == original_bytes
    assert record.filename == "survey.mp4"
    assert Path(record.stored_path) == stored.resolve()
    assert record.size_bytes == stored.stat().st_size
    assert record.frame_count == FRAME_COUNT
    assert record.fps == pytest.approx(VIDEO_FPS)
    assert record.duration_seconds == pytest.approx(FRAME_COUNT / VIDEO_FPS)
    assert (record.width, record.height) == VIDEO_SIZE

    persisted = get_video_by_filename("survey.mp4", db_path)
    assert persisted is not None
    assert persisted.video_id == record.video_id


def test_ingest_never_overwrites_on_collision(
    source_video: Path, raw_videos_dir: Path, db_path: Path
) -> None:
    first = ingest_video(source_video, raw_videos_dir=raw_videos_dir, db_path=db_path)
    second = ingest_video(source_video, raw_videos_dir=raw_videos_dir, db_path=db_path)

    assert (raw_videos_dir / "survey.mp4").is_file()
    assert (raw_videos_dir / "survey_1.mp4").is_file()
    assert first.filename == "survey.mp4"
    assert second.filename == "survey_1.mp4"
    assert first.video_id != second.video_id


def test_ingest_registers_existing_file_in_place(
    raw_videos_dir: Path, db_path: Path
) -> None:
    existing = write_synthetic_video(raw_videos_dir / "already_here.mp4")

    record = ingest_video(existing, raw_videos_dir=raw_videos_dir, db_path=db_path)

    assert record.filename == "already_here.mp4"
    # No duplicate copy was created inside the store.
    assert not (raw_videos_dir / "already_here_1.mp4").exists()
    # Re-registering the same file is idempotent (same row/video_id).
    again = ingest_video(existing, raw_videos_dir=raw_videos_dir, db_path=db_path)
    assert again.video_id == record.video_id


def test_ingest_rejects_unsupported_extension(
    tmp_path: Path, raw_videos_dir: Path, db_path: Path
) -> None:
    bad = tmp_path / "clip.avi"
    bad.write_bytes(b"not a real video")

    with pytest.raises(UnsupportedVideoFormatError):
        ingest_video(bad, raw_videos_dir=raw_videos_dir, db_path=db_path)


def test_ingest_missing_source_raises(
    tmp_path: Path, raw_videos_dir: Path, db_path: Path
) -> None:
    with pytest.raises(VideoNotFoundError):
        ingest_video(tmp_path / "ghost.mp4", raw_videos_dir=raw_videos_dir, db_path=db_path)


# ---------------------------------------------------------------------------
# Raw store resolution (path-traversal safety)
# ---------------------------------------------------------------------------
def test_resolve_rejects_path_traversal(tmp_path: Path, raw_videos_dir: Path) -> None:
    with pytest.raises(VideoNotFoundError):
        resolve_raw_video("../outside.mp4", raw_videos_dir=raw_videos_dir)
    with pytest.raises(VideoNotFoundError):
        resolve_raw_video(str(tmp_path / "absolute.mp4"), raw_videos_dir=raw_videos_dir)
    with pytest.raises(VideoNotFoundError):
        resolve_raw_video("missing.mp4", raw_videos_dir=raw_videos_dir)


def test_resolve_accepts_valid_video(raw_videos_dir: Path) -> None:
    video_path = write_synthetic_video(raw_videos_dir / "survey.mp4")

    resolved = resolve_raw_video("survey.mp4", raw_videos_dir=raw_videos_dir)

    assert resolved == video_path.resolve()


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------
def test_extract_frames_step_mode(raw_videos_dir: Path, frames_dir: Path) -> None:
    video_path = write_synthetic_video(raw_videos_dir / "survey.mp4")
    params = FrameExtractionRequest(filename="survey.mp4", frame_step=10)

    result = extract_frames(video_path, params, frames_dir=frames_dir, job_id="job123")

    run_dir = Path(result.output_dir)
    assert run_dir.parent == frames_dir / "survey"
    assert run_dir.name.startswith("run_")
    frame_files = sorted(run_dir.glob("frame_*.jpg"))
    assert [f.name for f in frame_files] == [
        "frame_000000.jpg",
        "frame_000010.jpg",
        "frame_000020.jpg",
    ]
    assert result.frames_written == 3
    assert result.total_video_frames == FRAME_COUNT
    assert result.processing_time_seconds > 0

    manifest = ExtractionManifest.model_validate_json(
        (run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.job_id == "job123"
    assert manifest.params.frame_step == 10
    assert manifest.frames_written == 3
    assert manifest.processing_time_seconds > 0
    assert manifest.opencv_version == cv2.__version__


def test_extract_frames_seconds_interval_mode(
    raw_videos_dir: Path, frames_dir: Path
) -> None:
    video_path = write_synthetic_video(raw_videos_dir / "survey.mp4")
    # 10 fps video → a frame every 1.0 s means indices 0, 10, 20.
    params = FrameExtractionRequest(filename="survey.mp4", seconds_interval=1.0)

    result = extract_frames(video_path, params, frames_dir=frames_dir)

    frame_files = sorted(Path(result.output_dir).glob("frame_*.jpg"))
    assert [f.name for f in frame_files] == [
        "frame_000000.jpg",
        "frame_000010.jpg",
        "frame_000020.jpg",
    ]
    assert result.frames_written == 3


def test_extract_frames_re_runs_never_overwrite(
    raw_videos_dir: Path, frames_dir: Path
) -> None:
    video_path = write_synthetic_video(raw_videos_dir / "survey.mp4")
    params = FrameExtractionRequest(filename="survey.mp4", frame_step=5)

    first = extract_frames(video_path, params, frames_dir=frames_dir)
    second = extract_frames(video_path, params, frames_dir=frames_dir)

    assert Path(first.output_dir) != Path(second.output_dir)
    assert Path(first.output_dir).is_dir()
    assert Path(second.output_dir).is_dir()


def test_extract_frames_missing_video_raises(frames_dir: Path) -> None:
    params = FrameExtractionRequest(filename="ghost.mp4", frame_step=1)

    with pytest.raises(VideoNotFoundError):
        extract_frames(Path("ghost.mp4"), params, frames_dir=frames_dir)


# ---------------------------------------------------------------------------
# Background job lifecycle (metrics logged to aerotwin.db)
# ---------------------------------------------------------------------------
def test_job_success_records_metrics(
    raw_videos_dir: Path, frames_dir: Path, db_path: Path
) -> None:
    write_synthetic_video(raw_videos_dir / "survey.mp4")
    params = FrameExtractionRequest(filename="survey.mp4", frame_step=10)

    video_path, job_id = prepare_extraction_job(
        "survey.mp4", params, raw_videos_dir=raw_videos_dir, db_path=db_path
    )
    run_frame_extraction_job(
        job_id, video_path, params, frames_dir=frames_dir, db_path=db_path
    )

    job = get_extraction_job(job_id, db_path)
    assert job is not None
    assert job.status == JobStatus.COMPLETED
    assert job.frames_written == 3
    assert job.total_video_frames == FRAME_COUNT
    assert job.processing_time_seconds is not None and job.processing_time_seconds > 0
    assert job.output_dir is not None and Path(job.output_dir).is_dir()
    assert job.error is None
    assert job.completed_at is not None
    # Sampling parameters round-trip from params_json (reproducibility log).
    assert job.params.frame_step == 10
    assert job.params.image_format == "jpg"


def test_job_failure_is_recorded_not_raised(
    raw_videos_dir: Path, frames_dir: Path, db_path: Path
) -> None:
    video_path = write_synthetic_video(raw_videos_dir / "survey.mp4")
    params = FrameExtractionRequest(filename="survey.mp4", frame_step=10)
    _, job_id = prepare_extraction_job(
        "survey.mp4", params, raw_videos_dir=raw_videos_dir, db_path=db_path
    )
    # Simulate the raw video disappearing between scheduling and execution.
    video_path.unlink()

    run_frame_extraction_job(
        job_id, video_path, params, frames_dir=frames_dir, db_path=db_path
    )

    job = get_extraction_job(job_id, db_path)
    assert job is not None
    assert job.status == JobStatus.FAILED
    assert job.error is not None and "not found" in job.error
    assert job.output_dir is None
    assert job.completed_at is not None


# ---------------------------------------------------------------------------
# API (FastAPI TestClient; background tasks run within the request cycle)
# ---------------------------------------------------------------------------
@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


def test_api_health_reports_wal(client: TestClient) -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["database"]["journal_mode"] == "wal"


def test_api_ingest_video_and_list(tmp_path: Path, client: TestClient) -> None:
    source = write_synthetic_video(tmp_path / "api_ingest.mp4")

    response = client.post(
        "/api/processing/ingest-video", json={"source_path": str(source)}
    )

    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "api_ingest.mp4"
    assert Path(body["stored_path"]).is_file()

    listing = client.get("/api/processing/videos")
    assert listing.status_code == 200
    assert any(video["filename"] == "api_ingest.mp4" for video in listing.json())


def test_api_ingest_missing_file_returns_404(tmp_path: Path, client: TestClient) -> None:
    response = client.post(
        "/api/processing/ingest-video",
        json={"source_path": str(tmp_path / "ghost.mp4")},
    )

    assert response.status_code == 404
    assert "detail" in response.json()


def test_api_ingest_rejects_bad_extension(tmp_path: Path, client: TestClient) -> None:
    bad = tmp_path / "notes.txt"
    bad.write_text("not a video", encoding="utf-8")

    response = client.post(
        "/api/processing/ingest-video", json={"source_path": str(bad)}
    )

    assert response.status_code == 415


def test_api_extract_frames_workflow(tmp_path: Path, client: TestClient) -> None:
    source = write_synthetic_video(tmp_path / "api_extract.mp4")
    ingest = client.post(
        "/api/processing/ingest-video", json={"source_path": str(source)}
    )
    assert ingest.status_code == 201

    response = client.post(
        "/api/processing/extract-frames",
        json={"filename": "api_extract.mp4", "frame_step": 10},
    )
    assert response.status_code == 202
    job_id = response.json()["job_id"]

    # The TestClient runs background tasks before returning, so the job is done.
    final = client.get(f"/api/processing/jobs/{job_id}")
    assert final.status_code == 200
    job = final.json()
    assert job["status"] == "completed"
    assert job["frames_written"] == 3
    assert job["error"] is None
    run_dir = Path(job["output_dir"])
    assert run_dir.is_dir()
    assert len(list(run_dir.glob("frame_*.jpg"))) == 3
    assert (run_dir / "manifest.json").is_file()

    jobs = client.get("/api/processing/jobs")
    assert jobs.status_code == 200
    assert any(item["job_id"] == job_id for item in jobs.json())


def test_api_extract_frames_unknown_video_returns_404(client: TestClient) -> None:
    response = client.post(
        "/api/processing/extract-frames",
        json={"filename": "nope.mp4", "frame_step": 5},
    )

    assert response.status_code == 404


def test_api_extract_frames_rejects_invalid_sampling(client: TestClient) -> None:
    both = client.post(
        "/api/processing/extract-frames",
        json={"filename": "x.mp4", "frame_step": 5, "seconds_interval": 1.0},
    )
    neither = client.post("/api/processing/extract-frames", json={"filename": "x.mp4"})

    assert both.status_code == 422
    assert neither.status_code == 422


def test_api_get_unknown_job_returns_404(client: TestClient) -> None:
    response = client.get("/api/processing/jobs/does-not-exist")

    assert response.status_code == 404
