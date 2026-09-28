"""Crack detection model wrapper (Phase 3: AI-Based Analysis).

Binary per-pixel crack segmentation (background vs. crack) with tiled
inference. Weights load from ``backend/app/models/weights/crack_model.pt``;
absent weights surface as a descriptive HTTP 503 through the API layer.
The mapping of detector output into crack maps and defect metrics is handled
by ``backend/app/services/crack_mapper.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .base import (
    MiniUNet,
    ModelWeightsMissingError,
    load_checkpoint,
    model_predict_proba,
    require_weights,
    resolve_device,
)

logger = logging.getLogger("aerotwin.models.crack")

CRACK_WEIGHTS_FILENAME: str = "crack_model.pt"
CRACK_CLASSES: tuple[str, ...] = ("background", "crack")
DEFAULT_TILE_SIZE: int = 256
DEFAULT_TILE_OVERLAP: int = 32
DEFAULT_CRACK_THRESHOLD: float = 0.5
JOB_TYPE_CRACK_MAPPING: str = "crack_mapping"


def ensure_model_available(*, weights_dir: Path | None = None) -> Path:
    """Raise :class:`ModelWeightsMissingError` (→ HTTP 503) unless weights exist."""
    return require_weights(CRACK_WEIGHTS_FILENAME, weights_dir=weights_dir)


class CrackDetector:
    """Binary crack segmentation with memory-safe tiled inference."""

    def __init__(
        self,
        weights_path: Path | None = None,
        *,
        tile_size: int = DEFAULT_TILE_SIZE,
        tile_overlap: int = DEFAULT_TILE_OVERLAP,
        threshold: float = DEFAULT_CRACK_THRESHOLD,
        device: str | None = None,
        base_channels: int = 16,
    ) -> None:
        if weights_path is None:
            weights_path = ensure_model_available()
        elif not weights_path.is_file():
            raise ModelWeightsMissingError(
                f"Model weights '{weights_path.name}' not found at {weights_path}. "
                "The API returns HTTP 503 until weights are present."
            )
        self.weights_path = weights_path
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.threshold = threshold
        self.device = resolve_device(device)
        self.num_classes = len(CRACK_CLASSES)
        self.model = MiniUNet(
            in_channels=3,
            num_classes=self.num_classes,
            base_channels=base_channels,
        )
        load_checkpoint(self.model, weights_path, self.device)
        logger.info(
            "CrackDetector ready | weights=%s device=%s tile=%d threshold=%.2f",
            weights_path.name,
            self.device,
            tile_size,
            threshold,
        )

    def predict_proba(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return ``(2, H, W)`` background/crack probabilities for a BGR frame."""
        return model_predict_proba(
            self.model,
            image_bgr,
            num_classes=self.num_classes,
            tile_size=self.tile_size,
            tile_overlap=self.tile_overlap,
            device=self.device,
        )

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return a ``(H, W)`` bool crack mask (crack probability ≥ threshold)."""
        probabilities = self.predict_proba(image_bgr)
        return probabilities[1] >= self.threshold

    def predict_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """Sequential per-frame inference keeps peak memory at one tile-batch."""
        return [self.predict(image) for image in images]
