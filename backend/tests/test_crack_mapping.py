"""Unit and integration tests for Phase 3 crack mapping & defect metrics.

Covers the metric math (Zhang-Suen length/width/area) on deterministic
synthetic masks, overlay rendering, and the end-to-end crack mapping job —
including DB metric rows in ``frame_defect_metrics`` — using fake weights.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.database import create_job_record, get_job_record, init_database, job_record_to_dict
from app.main import app
from app.models.base import MiniUNet
from app.models.crack_detector import CRACK_CLASSES, CRACK_WEIGHTS_FILENAME
from app.schemas.processing import (
    AnalysisRunManifest,
    CrackFrameMetrics,
    CrackMappingRequest,
    CrackRunSummary,
    JobStatus,
    ProcessingJobResponse,
)
from app.services.crack_mapper import render_crack_overlay, run_crack_mapping_job
from app.services.defect_metrics import (
    compute_crack_metrics,
    fetch_job_metrics,
    summarize_crack_metrics,
    zhang_suen_thinning,
)

FRAME_SHAPE: tuple[int, int] = (48, 64)  # h, w


def save_fake_crack_weights() -> Path:
    model = MiniUNet(in_channels=3, num_classes=len(CRACK_CLASSES), base_channels=16)
    path = settings.weights_dir / CRACK_WEIGHTS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path


def clear_crack_weights() -> None:
    (settings.weights_dir / CRACK_WEIGHTS_FILENAME).unlink(missing_ok=True)


def write_frame_run(run_name: str = "run_metrics", frame_count: int = 2) -> Path:
    run_dir = settings.frames_dir / "ph3_crack_video" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    for index in range(frame_count):
        frame = np.full((*FRAME_SHAPE, 3), (index * 40) % 255, dtype=np.uint8)
        assert cv2.imwrite(str(run_dir / f"frame_{index:06d}.jpg"), frame)
    return run_dir


def horizontal_crack_mask(height: int = 20, width: int = 100) -> np.ndarray:
    """Deterministic 3-px-wide, 80-px-long bar (rows 9-11, cols 10-89)."""
    mask = np.zeros((height, width), dtype=bool)
    mask[9:12, 10:90] = True
    return mask


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return init_database(tmp_path / "aerotwin.db")


# ---------------------------------------------------------------------------
# Defect metric math (deterministic synthetic masks)
# ---------------------------------------------------------------------------
def test_zhang_suen_thinning_keeps_length_drops_width() -> None:
    mask = horizontal_crack_mask()

    skeleton = zhang_suen_thinning(mask)

    # 80 px long, 3 px wide → 1 px wide skeleton of ~the same length.
    assert skeleton.sum() <= mask.sum()
    assert 75 <= skeleton.sum() <= 85
    assert int(skeleton.any(axis=1).sum()) <= 2


def test_compute_crack_metrics_for_bar() -> None:
    metrics = compute_crack_metrics("frame_000000.jpg", horizontal_crack_mask())

    assert metrics.crack_pixel_count == 240  # 80 x 3
    assert metrics.crack_area_ratio == pytest.approx(12.0)  # 240 / 2000 * 100
    assert 75 <= metrics.crack_length_px <= 85
    assert metrics.mean_width_px == pytest.approx(3.0, abs=0.4)
    assert metrics.component_count == 1


def test_compute_crack_metrics_empty_mask_is_zeroed() -> None:
    metrics = compute_crack_metrics("frame_000001.jpg", np.zeros((20, 100), dtype=bool))

    assert metrics.crack_pixel_count == 0
    assert metrics.crack_area_ratio == 0.0
    assert metrics.crack_length_px == 0.0
    assert metrics.mean_width_px == 0.0
    assert metrics.component_count == 0


def test_summarize_crack_metrics_aggregates() -> None:
    rows = [
        CrackFrameMetrics(
            frame_filename="a.jpg",
            crack_pixel_count=100,
            crack_area_ratio=1.0,
            crack_length_px=50.0,
            mean_width_px=2.0,
            component_count=1,
        ),
        CrackFrameMetrics(
            frame_filename="b.jpg",
            crack_pixel_count=0,
            crack_area_ratio=0.0,
            crack_length_px=0.0,
            mean_width_px=0.0,
            component_count=0,
        ),
        CrackFrameMetrics(
            frame_filename="c.jpg",
            crack_pixel_count=300,
            crack_area_ratio=3.0,
            crack_length_px=90.0,
            mean_width_px=3.0,
            component_count=2,
        ),
    ]

    summary = summarize_crack_metrics(rows)

    assert summary.frames_analyzed == 3
    assert summary.frames_with_cracks == 2
    assert summary.total_crack_pixels == 400
    assert summary.mean_area_ratio == pytest.approx(4 / 3)
    assert summary.mean_crack_length_px == pytest.approx(70.0)
    assert summary.mean_width_px == pytest.approx(2.5)
    assert summary.longest_crack_px == pytest.approx(90.0)


def test_render_crack_overlay_touches_only_mask_pixels() -> None:
    frame = np.full((*FRAME_SHAPE, 3), 100, dtype=np.uint8)
    mask = np.zeros(FRAME_SHAPE, dtype=bool)
    mask[10:15, 10:20] = True

    overlay = render_crack_overlay(frame, mask)

    assert overlay.shape == frame.shape and overlay.dtype == np.uint8
    np.testing.assert_array_equal(overlay[~mask], frame[~mask])
    assert np.any(overlay[mask] != frame[mask])  # cracks got highlighted


# ---------------------------------------------------------------------------
# Crack mapping job (outputs + metric rows in aerotwin.db)
# ---------------------------------------------------------------------------
def test_run_crack_mapping_job_end_to_end(
    tmp_path: Path, db_path: Path
) -> None:
    frames_run = write_frame_run("run_job")
    save_fake_crack_weights()
    request = CrackMappingRequest(frames_path=f"ph3_crack_video/{frames_run.name}")
    original = {path.name: path.read_bytes() for path in frames_run.glob("*.jpg")}
    job_id = create_job_record(
        job_type="crack_mapping",
        video_filename=frames_run.parent.name,
        params_json=request.model_dump_json(),
        db_path=db_path,
    )

    run_crack_mapping_job(
        job_id, frames_run, request, output_root=tmp_path / "cracks", db_path=db_path
    )

    job = ProcessingJobResponse.model_validate(
        job_record_to_dict(get_job_record(job_id, db_path))
    )
    assert job.status == JobStatus.COMPLETED
    assert job.frames_written == 2
    out_dir = Path(job.output_dir or "")
    assert len(list(out_dir.glob("crack_mask_*.png"))) == 2
    assert len(list(out_dir.glob("crack_overlay_*.jpg"))) == 2
    mask = cv2.imread(str(out_dir / "crack_mask_000000.png"), cv2.IMREAD_UNCHANGED)
    assert mask is not None and set(np.unique(mask)) <= {0, 255}
    # metrics.json: per-frame rows + summary; manifest for reproducibility.
    payload = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert len(payload["frames"]) == 2
    summary = CrackRunSummary.model_validate(payload["summary"])
    assert summary.frames_analyzed == 2
    manifest = AnalysisRunManifest.model_validate_json(
        (out_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.job_type == "crack_mapping"
    # Job-level summary + per-frame rows persisted in the DB.
    assert CrackRunSummary.model_validate(job.metrics).frames_analyzed == 2
    persisted = fetch_job_metrics(job_id, db_path)
    assert len(persisted) == 2
    assert {row.frame_filename for row in persisted} == {"frame_000000.jpg", "frame_000001.jpg"}
    # Raw frames are read-only inputs.
    assert {path.name: path.read_bytes() for path in frames_run.glob("*.jpg")} == original


def test_run_crack_mapping_job_records_missing_weights(tmp_path: Path, db_path: Path) -> None:
    frames_run = write_frame_run("run_fail")
    clear_crack_weights()
    request = CrackMappingRequest(frames_path=f"ph3_crack_video/{frames_run.name}")
    job_id = create_job_record(
        job_type="crack_mapping",
        video_filename=frames_run.parent.name,
        params_json=request.model_dump_json(),
        db_path=db_path,
    )

    run_crack_mapping_job(
        job_id, frames_run, request, output_root=tmp_path / "cracks", db_path=db_path
    )

    job = ProcessingJobResponse.model_validate(
        job_record_to_dict(get_job_record(job_id, db_path))
    )
    assert job.status == JobStatus.FAILED
    assert job.error is not None and CRACK_WEIGHTS_FILENAME in job.error
    assert job.output_dir is None
    assert not (tmp_path / "cracks").exists() or not any((tmp_path / "cracks").iterdir())


def test_api_map_cracks_full_workflow(client: TestClient) -> None:
    frames_run = write_frame_run("run_api")
    save_fake_crack_weights()

    queued = client.post(
        "/api/processing/map-cracks",
        json={"frames_path": f"ph3_crack_video/{frames_run.name}", "crack_threshold": 0.6},
    )

    assert queued.status_code == 202
    body = queued.json()
    assert body["job_type"] == "crack_mapping"
    # The TestClient runs background tasks before returning.
    job_id = body["job_id"]
    final = client.get(f"/api/processing/jobs/{job_id}")
    assert final.status_code == 200
    job = final.json()
    assert job["status"] == "completed"
    assert job["frames_written"] == 2
    assert job["metrics"]["frames_analyzed"] == 2
    assert Path(job["output_dir"]).is_dir()
    # The generic job list includes analysis jobs alongside extractions.
    listing = client.get("/api/processing/jobs")
    assert listing.status_code == 200
    assert any(item["job_id"] == job_id for item in listing.json())