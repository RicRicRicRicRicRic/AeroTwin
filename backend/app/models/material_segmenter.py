"""Material segmentation model wrapper (Phase 3: AI-Based Analysis).

Semantic segmentation of UAV frames into ``MATERIAL_CLASSES`` using a compact
U-Net backbone with tiled inference. Weights load from
``backend/app/models/weights/material_model.pt``; when absent, a descriptive
error propagates to the API which answers HTTP 503.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch

from ..core.config import settings
from ..core.database import update_job_record
from ..schemas.processing import AnalysisRunManifest, JobStatus, MaterialSegmentationRequest
from ..services.uav_preprocessor import list_run_frames
from .base import (
    MiniUNet,
    ModelWeightsMissingError,
    load_checkpoint,
    model_predict_proba,
    require_weights,
    resolve_device,
)

logger = logging.getLogger("aerotwin.models.material")

MATERIAL_WEIGHTS_FILENAME: str = "material_model.pt"
MATERIAL_CLASSES: tuple[str, ...] = (
    "background",
    "concrete",
    "metal",
    "glass",
    "brick",
    "roofing",
)
DEFAULT_TILE_SIZE: int = 256
DEFAULT_TILE_OVERLAP: int = 32
JOB_TYPE_MATERIAL_SEGMENTATION: str = "material_segmentation"


def ensure_model_available(*, weights_dir: Path | None = None) -> Path:
    """Raise :class:`ModelWeightsMissingError` (→ HTTP 503) unless weights exist."""
    return require_weights(MATERIAL_WEIGHTS_FILENAME, weights_dir=weights_dir)


class MaterialSegmenter:
    """Material segmentation over BGR frames with memory-safe tiled inference."""

    def __init__(
        self,
        weights_path: Path | None = None,
        *,
        tile_size: int = DEFAULT_TILE_SIZE,
        tile_overlap: int = DEFAULT_TILE_OVERLAP,
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
        self.device = resolve_device(device)
        self.num_classes = len(MATERIAL_CLASSES)
        self.model = MiniUNet(
            in_channels=3,
            num_classes=self.num_classes,
            base_channels=base_channels,
        )
        load_checkpoint(self.model, weights_path, self.device)
        logger.info(
            "MaterialSegmenter ready | weights=%s device=%s tile=%d",
            weights_path.name,
            self.device,
            tile_size,
        )

    def predict_proba(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return ``(C, H, W)`` class probabilities for a BGR frame."""
        return model_predict_proba(
            self.model,
            image_bgr,
            num_classes=self.num_classes,
            tile_size=self.tile_size,
            tile_overlap=self.tile_overlap,
            device=self.device,
        )

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return an ``(H, W)`` int64 map of :data:`MATERIAL_CLASSES` indices."""
        return np.argmax(self.predict_proba(image_bgr), axis=0).astype(np.int64)

    def predict_batch(self, images: Sequence[np.ndarray]) -> list[np.ndarray]:
        """Sequential per-frame inference keeps peak memory at one tile-batch."""
        return [self.predict(image) for image in images]


def run_material_segmentation_job(
    job_id: str,
    frames_run_dir: Path,
    params: MaterialSegmentationRequest,
    *,
    output_root: Path | None = None,
    db_path: Path | None = None,
) -> None:
    """Background task: segment every frame of a run into material masks.

    Writes ``mask_<frame_id>.png`` (uint8 class indices) plus a reproducibility
    ``manifest.json`` under ``data/processed/material_masks/<stem>/job_<id8>/``
    and records status/metrics in ``aerotwin.db``. Never raises — failures land
    on the job row; partial output directories are removed.
    """
    update_job_record(job_id, {"status": JobStatus.RUNNING.value}, db_path)
    started_at = datetime.now(timezone.utc)
    timer = time.perf_counter()
    root = output_root if output_root is not None else settings.material_masks_dir
    out_dir = root / frames_run_dir.parent.name / f"job_{job_id[:8]}"
    try:
        frames = list_run_frames(frames_run_dir)
        segmenter = MaterialSegmenter(tile_size=params.tile_size or DEFAULT_TILE_SIZE)
        out_dir.mkdir(parents=True, exist_ok=False)
        frame_times: list[float] = []
        for frame_path in frames:
            image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Unreadable frame image: {frame_path.name}")
            frame_timer = time.perf_counter()
            mask = segmenter.predict(image)
            frame_times.append(time.perf_counter() - frame_timer)
            mask_path = out_dir / f"mask_{frame_path.stem.removeprefix('frame_')}.png"
            if not cv2.imwrite(str(mask_path), mask.astype(np.uint8)):
                raise RuntimeError(f"Failed to write mask: {mask_path}")
        processing_time = time.perf_counter() - timer
        completed_at = datetime.now(timezone.utc)
        manifest = AnalysisRunManifest(
            job_id=job_id,
            job_type=JOB_TYPE_MATERIAL_SEGMENTATION,
            frames_run=frames_run_dir.name,
            frames_run_path=str(frames_run_dir),
            frames_processed=len(frames),
            params=params.model_dump(mode="json"),
            class_names=list(MATERIAL_CLASSES),
            weights_file=segmenter.weights_path.name,
            device=str(segmenter.device),
            tile_size=segmenter.tile_size,
            torch_version=torch.__version__,
            opencv_version=cv2.__version__,
            processing_time_seconds=processing_time,
            started_at=started_at,
            completed_at=completed_at,
        )
        (out_dir / "manifest.json").write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8"
        )
        metrics = {
            "frames_processed": len(frames),
            "mean_frame_inference_seconds": float(np.mean(frame_times)) if frame_times else 0.0,
            "total_inference_seconds": float(np.sum(frame_times)) if frame_times else 0.0,
            "device": str(segmenter.device),
            "class_names": list(MATERIAL_CLASSES),
        }
        update_job_record(
            job_id,
            {
                "status": JobStatus.COMPLETED.value,
                "output_dir": str(out_dir),
                "frames_written": len(frames),
                "total_video_frames": len(frames),
                "processing_time_seconds": processing_time,
                "metrics_json": json.dumps(metrics),
                "completed_at": completed_at.isoformat(),
            },
            db_path,
        )
        logger.info(
            "Material segmentation job %s finished: %d frames in %.3fs -> %s",
            job_id,
            len(frames),
            processing_time,
            out_dir,
        )
    except Exception as exc:  # noqa: BLE001 — job boundary: everything lands on the row
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)
        update_job_record(
            job_id,
            {
                "status": JobStatus.FAILED.value,
                "error": str(exc),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
            db_path,
        )
        logger.exception("Material segmentation job %s failed", job_id)
