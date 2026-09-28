"""Unit and API tests for Phase 3 crack detection model execution paths.

Weights are absent by default (isolated ``AEROTWIN_WEIGHTS_DIR``): missing
paths assert the descriptive HTTP-503 errors, while fake ``state_dict``
fixtures drive the real load → tiled inference → thresholding pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.models import crack_detector
from app.models.base import MiniUNet, ModelLoadError, ModelWeightsMissingError
from app.models.crack_detector import CRACK_CLASSES, CRACK_WEIGHTS_FILENAME, CrackDetector

FRAME_SHAPE: tuple[int, int] = (48, 64)  # h, w


def save_fake_crack_weights() -> Path:
    """Write a randomly initialised but structurally valid state_dict fixture."""
    model = MiniUNet(in_channels=3, num_classes=len(CRACK_CLASSES), base_channels=16)
    path = settings.weights_dir / CRACK_WEIGHTS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path


def clear_crack_weights() -> None:
    (settings.weights_dir / CRACK_WEIGHTS_FILENAME).unlink(missing_ok=True)


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# Missing / corrupt weights → descriptive errors and HTTP 503
# ---------------------------------------------------------------------------
def test_crack_weights_missing_raises_descriptive_error() -> None:
    clear_crack_weights()

    with pytest.raises(ModelWeightsMissingError) as excinfo:
        crack_detector.ensure_model_available()

    assert CRACK_WEIGHTS_FILENAME in str(excinfo.value)
    assert "HTTP 503" in str(excinfo.value)


def test_api_map_cracks_missing_weights_returns_503(client: TestClient) -> None:
    run_dir = settings.frames_dir / "ph3_crack_video" / "run_api"
    run_dir.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(
        str(run_dir / "frame_000000.jpg"), np.zeros((*FRAME_SHAPE, 3), dtype=np.uint8)
    )
    clear_crack_weights()

    response = client.post(
        "/api/processing/map-cracks",
        json={"frames_path": "ph3_crack_video/run_api"},
    )

    assert response.status_code == 503
    assert CRACK_WEIGHTS_FILENAME in response.json()["detail"]


def test_corrupt_weights_raise_model_load_error(tmp_path: Path) -> None:
    bad_weights = tmp_path / CRACK_WEIGHTS_FILENAME
    bad_weights.write_bytes(b"this is not a torch checkpoint")

    with pytest.raises(ModelLoadError) as excinfo:
        CrackDetector(weights_path=bad_weights)

    assert CRACK_WEIGHTS_FILENAME in str(excinfo.value)


# ---------------------------------------------------------------------------
# Model execution paths (fake state_dict fixtures)
# ---------------------------------------------------------------------------
def test_crack_detector_predict_returns_bool_mask() -> None:
    weights = save_fake_crack_weights()
    detector = CrackDetector(weights_path=weights, tile_size=64)
    rng = np.random.default_rng(3)
    image = rng.integers(0, 255, (*FRAME_SHAPE, 3), dtype=np.uint8)

    mask = detector.predict(image)

    assert mask.dtype == np.bool_
    assert mask.shape == FRAME_SHAPE


def test_crack_detector_tiled_probabilities_are_valid() -> None:
    weights = save_fake_crack_weights()
    detector = CrackDetector(weights_path=weights, tile_size=32, tile_overlap=8)
    rng = np.random.default_rng(5)
    image = rng.integers(0, 255, (70, 90, 3), dtype=np.uint8)

    probabilities = detector.predict_proba(image)

    assert probabilities.shape == (2, 70, 90)
    np.testing.assert_allclose(probabilities.sum(axis=0), 1.0, atol=1e-3)


def test_crack_threshold_is_monotonic() -> None:
    weights = save_fake_crack_weights()
    rng = np.random.default_rng(9)
    image = rng.integers(0, 255, (*FRAME_SHAPE, 3), dtype=np.uint8)

    permissive = CrackDetector(weights_path=weights, tile_size=64, threshold=0.1).predict(image)
    strict = CrackDetector(weights_path=weights, tile_size=64, threshold=0.9).predict(image)

    # A stricter threshold can only shrink the mask (never adds pixels).
    assert np.all(~strict | permissive)