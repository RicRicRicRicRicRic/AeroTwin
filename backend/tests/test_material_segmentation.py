"""Unit and API tests for Phase 3: material segmentation & element detection.

Weights are absent from the test environment by default (isolated via
``AEROTWIN_WEIGHTS_DIR`` in conftest). Missing-weight paths are asserted
directly; execution paths run for real against fake ``state_dict`` fixtures
written on demand, covering load → tiled inference → mask output → DB rows.
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
from app.models import material_segmenter, structural_element_detector
from app.models.base import MiniUNet, ModelWeightsMissingError, TileBlender
from app.models.material_segmenter import (
    MATERIAL_CLASSES,
    MATERIAL_WEIGHTS_FILENAME,
    MaterialSegmenter,
    run_material_segmentation_job,
)
from app.models.structural_element_detector import (
    ELEMENT_CLASSES,
    ELEMENT_WEIGHTS_FILENAME,
    StructuralElementDetector,
    run_element_detection_job,
)
from app.schemas.processing import (
    AnalysisRunManifest,
    ElementDetectionRequest,
    JobStatus,
    MaterialSegmentationRequest,
    ProcessingJobResponse,
)

FRAME_SHAPE: tuple[int, int] = (48, 64)  # h, w


def save_fake_weights(filename: str, *, num_classes: int) -> Path:
    """Write a randomly initialised but structurally valid state_dict fixture."""
    model = MiniUNet(in_channels=3, num_classes=num_classes, base_channels=16)
    path = settings.weights_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path


def clear_weights(filename: str) -> None:
    (settings.weights_dir / filename).unlink(missing_ok=True)


def write_frame_run(run_name: str = "run_unit", frame_count: int = 2) -> Path:
    """Create a frames run under the (test) frames dir with synthetic frames."""
    run_dir = settings.frames_dir / "ph3_video" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    for index in range(frame_count):
        frame = np.full((*FRAME_SHAPE, 3), (index * 40) % 255, dtype=np.uint8)
        assert cv2.imwrite(str(run_dir / f"frame_{index:06d}.jpg"), frame)
    return run_dir


@pytest.fixture
def frames_run() -> Path:
    return write_frame_run()


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return init_database(tmp_path / "aerotwin.db")


# ---------------------------------------------------------------------------
# Missing weights → descriptive errors / HTTP 503
# ---------------------------------------------------------------------------
def test_material_weights_missing_raises_descriptive_error() -> None:
    clear_weights(MATERIAL_WEIGHTS_FILENAME)

    with pytest.raises(ModelWeightsMissingError) as excinfo:
        material_segmenter.ensure_model_available()

    assert MATERIAL_WEIGHTS_FILENAME in str(excinfo.value)
    assert "HTTP 503" in str(excinfo.value)


def test_api_segment_materials_missing_weights_returns_503(
    frames_run: Path, client: TestClient
) -> None:
    clear_weights(MATERIAL_WEIGHTS_FILENAME)

    response = client.post(
        "/api/processing/segment-materials",
        json={"frames_path": f"ph3_video/{frames_run.name}"},
    )

    assert response.status_code == 503
    assert MATERIAL_WEIGHTS_FILENAME in response.json()["detail"]


def test_api_detect_elements_missing_weights_returns_503(
    frames_run: Path, client: TestClient
) -> None:
    clear_weights(ELEMENT_WEIGHTS_FILENAME)

    response = client.post(
        "/api/processing/detect-elements",
        json={"frames_path": f"ph3_video/{frames_run.name}"},
    )

    assert response.status_code == 503
    assert ELEMENT_WEIGHTS_FILENAME in response.json()["detail"]


def test_api_analysis_unknown_frames_run_returns_404(client: TestClient) -> None:
    response = client.post(
        "/api/processing/segment-materials",
        json={"frames_path": "no_such_video/run_nope"},
    )

    assert response.status_code == 404


def test_api_analysis_rejects_path_escape(client: TestClient) -> None:
    response = client.post(
        "/api/processing/segment-materials",
        json={"frames_path": "../../outside"},
    )

    assert response.status_code == 400


# ---------------------------------------------------------------------------
# Model execution paths (fake state_dict fixtures)
# ---------------------------------------------------------------------------
def test_segmenter_predict_with_fake_weights() -> None:
    weights = save_fake_weights(MATERIAL_WEIGHTS_FILENAME, num_classes=len(MATERIAL_CLASSES))
    segmenter = MaterialSegmenter(weights_path=weights, tile_size=64)
    rng = np.random.default_rng(42)
    image = rng.integers(0, 255, (*FRAME_SHAPE, 3), dtype=np.uint8)

    mask = segmenter.predict(image)

    assert mask.shape == FRAME_SHAPE
    assert 0 <= mask.min() and mask.max() < len(MATERIAL_CLASSES)


def test_segmenter_tiled_inference_produces_valid_probabilities() -> None:
    weights = save_fake_weights(MATERIAL_WEIGHTS_FILENAME, num_classes=len(MATERIAL_CLASSES))
    segmenter = MaterialSegmenter(weights_path=weights, tile_size=32, tile_overlap=8)
    rng = np.random.default_rng(7)
    image = rng.integers(0, 255, (90, 100, 3), dtype=np.uint8)

    probabilities = segmenter.predict_proba(image)

    assert probabilities.shape == (len(MATERIAL_CLASSES), 90, 100)
    np.testing.assert_allclose(probabilities.sum(axis=0), 1.0, atol=1e-3)
    assert segmenter.predict(image).shape == (90, 100)


def test_tile_blender_cross_fades_overlap_seam() -> None:
    blender = TileBlender(1, 1, 4, ramp=2)
    blender.add(np.full((1, 1, 3), 1.0, dtype=np.float32), 0, 0)
    blender.add(np.full((1, 1, 3), 3.0, dtype=np.float32), 0, 1)

    row = blender.finalize()[0, 0]

    # Single-source borders, a smooth monotonic cross-fade across the seam.
    np.testing.assert_allclose(row, [1.0, 5 / 3, 7 / 3, 3.0], atol=1e-5)
    assert np.all(np.diff(row) > 0)


# ---------------------------------------------------------------------------
# Job runners (status, outputs, metrics in aerotwin.db)
# ---------------------------------------------------------------------------
def test_run_material_segmentation_job_writes_masks(
    frames_run: Path, tmp_path: Path, db_path: Path
) -> None:
    save_fake_weights(MATERIAL_WEIGHTS_FILENAME, num_classes=len(MATERIAL_CLASSES))
    request = MaterialSegmentationRequest(frames_path=f"ph3_video/{frames_run.name}")
    original = {path.name: path.read_bytes() for path in frames_run.glob("*.jpg")}
    job_id = create_job_record(
        job_type=material_segmenter.JOB_TYPE_MATERIAL_SEGMENTATION,
        video_filename=frames_run.parent.name,
        params_json=request.model_dump_json(),
        db_path=db_path,
    )

    run_material_segmentation_job(
        job_id, frames_run, request, output_root=tmp_path / "masks", db_path=db_path
    )

    job = ProcessingJobResponse.model_validate(
        job_record_to_dict(get_job_record(job_id, db_path))
    )
    assert job.status == JobStatus.COMPLETED
    assert job.frames_written == 2
    assert job.error is None
    out_dir = Path(job.output_dir or "")
    masks = sorted(out_dir.glob("mask_*.png"))
    assert [path.name for path in masks] == ["mask_000000.png", "mask_000001.png"]
    mask = cv2.imread(str(masks[0]), cv2.IMREAD_UNCHANGED)
    assert mask is not None and mask.shape == FRAME_SHAPE
    assert mask.max() < len(MATERIAL_CLASSES)
    manifest = AnalysisRunManifest.model_validate_json(
        (out_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.job_type == material_segmenter.JOB_TYPE_MATERIAL_SEGMENTATION
    assert manifest.frames_processed == 2
    assert manifest.class_names == list(MATERIAL_CLASSES)
    metrics = job.metrics or {}
    assert metrics["frames_processed"] == 2
    # Raw frames are read-only inputs: contents must be byte-identical.
    assert {path.name: path.read_bytes() for path in frames_run.glob("*.jpg")} == original


def test_run_material_segmentation_job_records_missing_weights(
    frames_run: Path, tmp_path: Path, db_path: Path
) -> None:
    clear_weights(MATERIAL_WEIGHTS_FILENAME)
    request = MaterialSegmentationRequest(frames_path=f"ph3_video/{frames_run.name}")
    job_id = create_job_record(
        job_type=material_segmenter.JOB_TYPE_MATERIAL_SEGMENTATION,
        video_filename=frames_run.parent.name,
        params_json=request.model_dump_json(),
        db_path=db_path,
    )

    run_material_segmentation_job(
        job_id, frames_run, request, output_root=tmp_path / "masks", db_path=db_path
    )

    job = ProcessingJobResponse.model_validate(
        job_record_to_dict(get_job_record(job_id, db_path))
    )
    assert job.status == JobStatus.FAILED
    assert job.error is not None and MATERIAL_WEIGHTS_FILENAME in job.error
    assert job.output_dir is None
    assert not (tmp_path / "masks").exists() or not any((tmp_path / "masks").iterdir())


def test_run_element_detection_job_writes_masks_and_boxes(
    frames_run: Path, tmp_path: Path, db_path: Path
) -> None:
    save_fake_weights(ELEMENT_WEIGHTS_FILENAME, num_classes=len(ELEMENT_CLASSES))
    request = ElementDetectionRequest(frames_path=f"ph3_video/{frames_run.name}")
    job_id = create_job_record(
        job_type=structural_element_detector.JOB_TYPE_ELEMENT_DETECTION,
        video_filename=frames_run.parent.name,
        params_json=request.model_dump_json(),
        db_path=db_path,
    )

    run_element_detection_job(
        job_id, frames_run, request, output_root=tmp_path / "elements", db_path=db_path
    )

    job = ProcessingJobResponse.model_validate(
        job_record_to_dict(get_job_record(job_id, db_path))
    )
    assert job.status == JobStatus.COMPLETED
    assert job.frames_written == 2
    out_dir = Path(job.output_dir or "")
    assert len(list(out_dir.glob("mask_*.png"))) == 2
    boxes = json.loads((out_dir / "elements.json").read_text(encoding="utf-8"))
    assert len(boxes["frames"]) == 2
    assert (out_dir / "manifest.json").is_file()
    metrics = job.metrics or {}
    assert metrics["frames_processed"] == 2
    assert "elements_by_label" in metrics


def test_element_detector_detect_returns_boxes() -> None:
    weights = save_fake_weights(ELEMENT_WEIGHTS_FILENAME, num_classes=len(ELEMENT_CLASSES))
    detector = StructuralElementDetector(weights_path=weights, tile_size=64)
    rng = np.random.default_rng(11)
    image = rng.integers(0, 255, (*FRAME_SHAPE, 3), dtype=np.uint8)

    mask, elements = detector.detect(image)

    assert mask.shape == FRAME_SHAPE
    assert 0 <= mask.min() and mask.max() < len(ELEMENT_CLASSES)
    for element in elements:
        assert element.label in ELEMENT_CLASSES[1:]
        assert 0.0 <= element.confidence <= 1.0
        assert element.width > 0 and element.height > 0
