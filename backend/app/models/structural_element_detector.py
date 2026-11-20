"""Structural element detection model wrapper (Phase 3: AI-Based Analysis).

Detects structural elements (window, door, column, wall, beam) as instance
segmentation masks using an Ultralytics YOLOv8n-seg checkpoint. Weights load
from ``backend/app/models/weights/element_model.pt``; absent weights surface as
a descriptive HTTP 503 through the API layer.

The checkpoint is *not* a TorchScript model.  It is loaded with
``ultralytics.YOLO`` and inference uses ``model.predict()``.  Instance masks
from ``result.masks.data`` are combined with their detection class IDs from
``result.boxes.cls`` to produce a per-frame class-index mask.

The actual checkpoint classes are:
    0 → window, 1 → door, 2 → column, 3 → wall, 4 → beam

Categories such as slab, shear_wall, and masonry_infill are *not* among the
model's trained classes and must not be reported as detections.
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
    ModelWeightsMissingError,
    require_weights,
    resolve_device,
)

logger = logging.getLogger("aerotwin.models.elements")

ELEMENT_WEIGHTS_FILENAME: str = "element_model.pt"

# Actual checkpoint classes (class ID → name).
ELEMENT_CLASSES: tuple[str, ...] = (
    "background",
    "window",
    "door",
    "column",
    "wall",
    "beam",
)

# Ultralytics class index → local class index (0 = background).
# Ultralytics: 0=window, 1=door, 2=column, 3=wall, 4=beam.
ELEMENT_CLASS_MAP: dict[int, int] = {
    0: 1,  # window → local 1
    1: 2,  # door → local 2
    2: 3,  # column → local 3
    3: 4,  # wall → local 4
    4: 5,  # beam → local 5
}

DEFAULT_TILE_SIZE: int = 640
DEFAULT_TILE_OVERLAP: int = 32
#: Components smaller than this many pixels are treated as noise.
MIN_COMPONENT_AREA: int = 16
JOB_TYPE_ELEMENT_DETECTION: str = "element_detection"


def ensure_model_available(*, weights_dir: Path | None = None) -> Path:
    """Raise :class:`ModelWeightsMissingError` (→ HTTP 503) unless weights exist."""
    return require_weights(ELEMENT_WEIGHTS_FILENAME, weights_dir=weights_dir)


class StructuralElementDetector:
    """Element instance-segmentation + bounding-box detection via Ultralytics YOLO."""

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

        # Load with Ultralytics YOLO — NOT TorchScript, NOT MiniUNet.
        try:
            from ultralytics import YOLO

            logger.info(
                "Loading YOLOv8-seg model from %s", self.weights_path
            )
            self.model = YOLO(str(self.weights_path))
            self.model.to(self.device)
            logger.info(
                "YOLO model loaded | weights=%s task=%s names=%s device=%s",
                self.weights_path.name,
                getattr(self.model.model, "task", "unknown"),
                self.model.names,
                self.device,
            )
        except ImportError as exc:
            raise ModelWeightsMissingError(
                "Ultralytics checkpoints require the ``ultralytics`` package; "
                "install it with ``pip install ultralytics``."
            ) from exc

        logger.info(
            "StructuralElementDetector ready | weights=%s device=%s tile=%d",
            weights_path.name,
            self.device,
            tile_size,
        )

    def detect(self, image_bgr: np.ndarray) -> tuple[np.ndarray, list[DetectedElement]]:
        """Run YOLOv8-seg inference and return a class-index mask + detections.

        Returns:
            mask: ``(H, W)`` int64 array with local class indices (0-5).
            elements: list of :class:`DetectedElement` for each instance.
        """
        h, w = image_bgr.shape[:2]
        logger.debug(
            "Detecting | frame_size=(%d, %d) conf=0.25 imgsz=%d", w, h, self.tile_size
        )

        results = self.model.predict(
            source=image_bgr,
            conf=0.25,
            imgsz=self.tile_size,
            verbose=False,
        )

        result = results[0]

        # Log detection summary.
        num_boxes = 0 if result.boxes is None else len(result.boxes)
        masks_exist = result.masks is not None and result.masks.data is not None
        logger.info(
            "Detections | num_instances=%d masks_exist=%s",
            num_boxes, masks_exist,
        )

        # Build class-index mask from instance masks.
        mask = np.zeros((h, w), dtype=np.int64)
        elements: list[DetectedElement] = []

        if result.boxes is None or num_boxes == 0:
            logger.warning("No detections in frame — mask is all background (0).")
            return mask, elements

        if not masks_exist:
            logger.error(
                "result.masks is None — mask extraction failed. "
                "This may indicate the checkpoint is not a segmentation model."
            )
            return mask, elements

        # Extract per-instance data.
        boxes = result.boxes
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        mask_data = result.masks.data.cpu().numpy()  # (num_instances, H_inf, W_inf)

        logger.debug(
            "Instance masks | shape=%s dtype=%s class_ids=%s confs=%s",
            mask_data.shape, mask_data.dtype, cls_ids.tolist(),
            np.round(confs, 3).tolist(),
        )

        # Resize instance masks to original frame dimensions if needed.
        if mask_data.shape[-2:] != (h, w):
            mask_data_resized = np.stack([
                cv2.resize(
                    m.astype(np.float32), (w, h),
                    interpolation=cv2.INTER_NEAREST,
                ) > 0.5
                for m in mask_data
            ])
        else:
            mask_data_resized = mask_data > 0.5

        # Combine instance masks: highest-confidence instance wins per pixel.
        # Sort by confidence ascending so higher-confidence instances overwrite.
        order = np.argsort(confs)
        foreground_pixels = np.zeros((h, w), dtype=bool)
        class_counts: dict[int, int] = {cid: 0 for cid in range(1, self.num_classes)}

        for idx in order:
            cls_id = int(cls_ids[idx])
            conf = float(confs[idx])
            inst_mask = mask_data_resized[idx]

            local_idx = ELEMENT_CLASS_MAP.get(cls_id, 0)
            if local_idx == 0:
                # Class not in class map (e.g. window/door not mapped) → skip
                logger.debug(
                    "Skipping class %d (not in ELEMENT_CLASS_MAP)", cls_id
                )
                continue

            # Only write where this instance has foreground and no higher-conf
            # instance has already claimed the pixel.
            new_pixels = inst_mask & ~foreground_pixels
            mask[new_pixels] = local_idx
            foreground_pixels |= inst_mask

            class_counts[local_idx] = class_counts.get(local_idx, 0) + int(new_pixels.sum())

            # Extract bounding box for this instance.
            x1, y1, x2, y2 = boxes.xyxy[idx].cpu().numpy().astype(int)
            elements.append(
                DetectedElement(
                    label=ELEMENT_CLASSES[local_idx],
                    x=int(x1),
                    y=int(y1),
                    width=int(x2 - x1),
                    height=int(y2 - y1),
                    confidence=round(conf, 4),
                )
            )

        # Log per-class foreground pixel counts.
        total = h * w
        for local_idx in range(1, self.num_classes):
            count = class_counts.get(local_idx, 0)
            pct = count / total * 100 if total > 0 else 0.0
            logger.info(
                "Class %s (%d): %d pixels (%.2f%%)",
                ELEMENT_CLASSES[local_idx], local_idx, count, pct,
            )

        logger.info(
            "Saved mask | shape=%s unique=%s min=%d max=%d",
            mask.shape, np.unique(mask).tolist(), int(mask.min()), int(mask.max()),
        )

        return mask, elements

    def predict_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """Sequential per-frame inference."""
        return [self.detect(image)[0] for image in images]


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
