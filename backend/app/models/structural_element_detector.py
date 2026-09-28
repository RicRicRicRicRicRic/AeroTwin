"""Structural element detection model wrapper (Phase 3: AI-Based Analysis).

Detects load-bearing elements (columns, beams, walls) as semantic masks and
derives per-element bounding boxes via connected components. Weights load from
``backend/app/models/weights/element_model.pt``; absent weights surface as a
descriptive HTTP 503 through the API layer.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

from ..core.config import settings
from ..core.database import update_job_record
from ..schemas.processing import (
    AnalysisRunManifest,
    DetectedElement,
    ElementDetectionRequest,
    JobStatus,
)
from ..services.uav_preprocessor import list_run_frames
from .base import (
    MiniUNet,
    ModelWeightsMissingError,
    load_checkpoint,
    model_predict_proba,
    require_weights,
    resolve_device,
)

logger = logging.getLogger("aerotwin.models.elements")

ELEMENT_WEIGHTS_FILENAME: str = "element_model.pt"
ELEMENT_CLASSES: tuple[str, ...] = ("background", "column", "beam", "wall")
DEFAULT_TILE_SIZE: int = 256
DEFAULT_TILE_OVERLAP: int = 32
#: Components smaller than this many pixels are treated as noise.
MIN_COMPONENT_AREA: int = 16
JOB_TYPE_ELEMENT_DETECTION: str = "element_detection"


def ensure_model_available(*, weights_dir: Path | None = None) -> Path:
    """Raise :class:`ModelWeightsMissingError` (→ HTTP 503) unless weights exist."""
    return require_weights(ELEMENT_WEIGHTS_FILENAME, weights_dir=weights_dir)


class StructuralElementDetector:
    """Element mask + bounding-box detection with tiled inference."""

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
        self.num_classes = len(ELEMENT_CLASSES)
        self.model = MiniUNet(
            in_channels=3,
            num_classes=self.num_classes,
            base_channels=base_channels,
        )
        load_checkpoint(self.model, weights_path, self.device)
        logger.info(
            "StructuralElementDetector ready | weights=%s device=%s tile=%d",
            weights_path.name,
            self.device,
            tile_size,
        )

    def predict_proba(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return ``(C, H, W)`` element-class probabilities for a BGR frame."""
        return model_predict_proba(
            self.model,
            image_bgr,
            num_classes=self.num_classes,
            tile_size=self.tile_size,
            tile_overlap=self.tile_overlap,
            device=self.device,
        )

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return an ``(H, W)`` int64 map of :data:`ELEMENT_CLASSES` indices."""
        return np.argmax(self.predict_proba(image_bgr), axis=0).astype(np.int64)

    def detect(self, image_bgr: np.ndarray) -> tuple[np.ndarray, list[DetectedElement]]:
        """Segment the frame and extract labelled bounding boxes.

        Each non-background class mask is split with connected components;
        boxes below :data:`MIN_COMPONENT_AREA` pixels are dropped and the
        confidence is the mean class probability inside the component.
        """
        probabilities = self.predict_proba(image_bgr)
        mask = np.argmax(probabilities, axis=0).astype(np.int64)
        elements: list[DetectedElement] = []
        for class_index, label in enumerate(ELEMENT_CLASSES[1:], start=1):
            class_mask = (mask == class_index).astype(np.uint8)
            if not class_mask.any():
                continue
            count, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=8)
            for component in range(1, count):
                x, y, width, height, area = (int(value) for value in stats[component])
                if area < MIN_COMPONENT_AREA:
                    continue
                confidence = float(probabilities[class_index][labels == component].mean())
                elements.append(
                    DetectedElement(
                        label=label,
                        x=x,
                        y=y,
                        width=width,
                        height=height,
                        confidence=round(min(max(confidence, 0.0), 1.0), 4),
                    )
                )
        return mask, elements


def run_element_detection_job(
    job_id: str,
    frames_run_dir: Path,
    params: ElementDetectionRequest,
    *,
    output_root: Path | None = None,
    db_path: Path | None = None,
) -> None:
    """Background task: detect structural elements in every frame of a run.

    Writes ``mask_<frame_id>.png`` (uint8 class indices), ``elements.json``
    (per-frame bounding boxes), and a reproducibility ``manifest.json`` under
    ``data/processed/structural_elements/<stem>/job_<id8>/``; status/metrics go
    to ``aerotwin.db``. Never raises — failures land on the job row.
    """
    update_job_record(job_id, {"status": JobStatus.RUNNING.value}, db_path)
    started_at = datetime.now(timezone.utc)
    timer = time.perf_counter()
    root = output_root if output_root is not None else settings.structural_elements_dir
    out_dir = root / frames_run_dir.parent.name / f"job_{job_id[:8]}"
    try:
        frames = list_run_frames(frames_run_dir)
        detector = StructuralElementDetector()
        out_dir.mkdir(parents=True, exist_ok=False)
        frame_times: list[float] = []
        detections: list[dict[str, object]] = []
        element_counts: dict[str, int] = {label: 0 for label in ELEMENT_CLASSES[1:]}
        for frame_path in frames:
            image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Unreadable frame image: {frame_path.name}")
            frame_timer = time.perf_counter()
            mask, elements = detector.detect(image)
            frame_times.append(time.perf_counter() - frame_timer)
            mask_path = out_dir / f"mask_{frame_path.stem.removeprefix('frame_')}.png"
            if not cv2.imwrite(str(mask_path), mask.astype(np.uint8)):
                raise RuntimeError(f"Failed to write mask: {mask_path}")
            detections.append(
                {"frame": frame_path.name, "elements": [element.model_dump() for element in elements]}
            )
            for element in elements:
                element_counts[element.label] = element_counts.get(element.label, 0) + 1
        processing_time = time.perf_counter() - timer
        completed_at = datetime.now(timezone.utc)
        (out_dir / "elements.json").write_text(
            json.dumps({"frames": detections}, indent=2),
            encoding="utf-8",
        )
        manifest = AnalysisRunManifest(
            job_id=job_id,
            job_type=JOB_TYPE_ELEMENT_DETECTION,
            frames_run=frames_run_dir.name,
            frames_run_path=str(frames_run_dir),
            frames_processed=len(frames),
            params=params.model_dump(mode="json"),
            class_names=list(ELEMENT_CLASSES),
            weights_file=detector.weights_path.name,
            device=str(detector.device),
            tile_size=detector.tile_size,
            torch_version=torch.__version__,
            opencv_version=cv2.__version__,
            processing_time_seconds=processing_time,
            started_at=started_at,
            completed_at=completed_at,
        )
        (out_dir / "manifest.json").write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8"
        )
        total_elements = sum(element_counts.values())
        metrics = {
            "frames_processed": len(frames),
            "total_elements": total_elements,
            "elements_by_label": element_counts,
            "mean_frame_inference_seconds": float(np.mean(frame_times)) if frame_times else 0.0,
            "device": str(detector.device),
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
            "Element detection job %s finished: %d frames, %d elements in %.3fs -> %s",
            job_id,
            len(frames),
            total_elements,
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
        logger.exception("Element detection job %s failed", job_id)
