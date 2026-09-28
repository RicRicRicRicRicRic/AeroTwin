"""Crack mapping service (Phase 3): detector output → mapped defect artifacts.

Orchestrates :class:`~app.models.crack_detector.CrackDetector` over an
extracted-frames run and writes, per run, under
``data/processed/crack_maps/<stem>/job_<id8>/``:

* ``crack_mask_<id>.png``    — binary crack masks (0/255)
* ``crack_overlay_<id>.jpg`` — frames with a translucent red crack overlay
* ``metrics.json``           — per-frame defect metrics + run summary
* ``manifest.json``          — reproducibility sidecar

Per-frame metric rows are persisted to ``aerotwin.db``
(``frame_defect_metrics``). Raw inputs are never modified and unique job
directories guarantee outputs are never overwritten.
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
from ..models.crack_detector import CrackDetector, JOB_TYPE_CRACK_MAPPING
from ..schemas.processing import (
    AnalysisRunManifest,
    CrackMappingRequest,
    CrackRunSummary,
    JobStatus,
)
from ..schemas.processing import CrackFrameMetrics
from .defect_metrics import compute_crack_metrics, log_frame_metrics, summarize_crack_metrics
from .uav_preprocessor import list_run_frames

logger = logging.getLogger("aerotwin.crack_mapper")

#: Crack overlay styling (BGR) applied to visualized defect maps.
OVERLAY_COLOR_BGR: tuple[int, int, int] = (0, 0, 255)
OVERLAY_ALPHA: float = 0.35


def render_crack_overlay(frame_bgr: np.ndarray, crack_mask: np.ndarray) -> np.ndarray:
    """Blend a translucent red overlay of *crack_mask* onto *frame_bgr*.

    Raises:
        ValueError: Frame and mask shapes disagree.
    """
    if frame_bgr.shape[:2] != np.asarray(crack_mask).shape[:2]:
        raise ValueError(
            f"Shape mismatch: frame {frame_bgr.shape[:2]} vs mask {np.asarray(crack_mask).shape[:2]}"
        )
    overlay = frame_bgr.astype(np.float32)
    mask = np.asarray(crack_mask, dtype=bool)
    overlay[mask] = (
        (1.0 - OVERLAY_ALPHA) * overlay[mask]
        + OVERLAY_ALPHA * np.asarray(OVERLAY_COLOR_BGR, dtype=np.float32)
    )
    return np.clip(overlay, 0, 255).astype(np.uint8)


def run_crack_mapping_job(
    job_id: str,
    frames_run_dir: Path,
    params: CrackMappingRequest,
    *,
    output_root: Path | None = None,
    db_path: Path | None = None,
) -> None:
    """Background task: detect cracks, map them, and log defect metrics.

    Never raises — any failure is recorded on the job row and partial output
    directories are removed so incomplete runs can't be mistaken for results.
    """
    update_job_record(job_id, {"status": JobStatus.RUNNING.value}, db_path)
    started_at = datetime.now(timezone.utc)
    timer = time.perf_counter()
    root = output_root if output_root is not None else settings.crack_maps_dir
    out_dir = root / frames_run_dir.parent.name / f"job_{job_id[:8]}"
    try:
        frames = list_run_frames(frames_run_dir)
        detector = CrackDetector(threshold=params.crack_threshold)
        out_dir.mkdir(parents=True, exist_ok=False)
        frame_times: list[float] = []
        frame_metrics: list[CrackFrameMetrics] = []
        for frame_path in frames:
            image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Unreadable frame image: {frame_path.name}")
            frame_timer = time.perf_counter()
            crack_mask = detector.predict(image)
            frame_times.append(time.perf_counter() - frame_timer)
            frame_id = frame_path.stem.removeprefix("frame_")
            mask_path = out_dir / f"crack_mask_{frame_id}.png"
            if not cv2.imwrite(str(mask_path), crack_mask.astype(np.uint8) * 255):
                raise RuntimeError(f"Failed to write crack mask: {mask_path}")
            overlay_path = out_dir / f"crack_overlay_{frame_id}.jpg"
            overlay = render_crack_overlay(image, crack_mask)
            if not cv2.imwrite(str(overlay_path), overlay):
                raise RuntimeError(f"Failed to write crack overlay: {overlay_path}")
            frame_metrics.append(compute_crack_metrics(frame_path.name, crack_mask))
        summary = summarize_crack_metrics(frame_metrics)
        log_frame_metrics(job_id, frame_metrics, db_path)
        processing_time = time.perf_counter() - timer
        completed_at = datetime.now(timezone.utc)
        (out_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "summary": summary.model_dump(mode="json"),
                    "frames": [row.model_dump(mode="json") for row in frame_metrics],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        manifest = AnalysisRunManifest(
            job_id=job_id,
            job_type=JOB_TYPE_CRACK_MAPPING,
            frames_run=frames_run_dir.name,
            frames_run_path=str(frames_run_dir),
            frames_processed=len(frames),
            params=params.model_dump(mode="json"),
            class_names=None,
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
        update_job_record(
            job_id,
            {
                "status": JobStatus.COMPLETED.value,
                "output_dir": str(out_dir),
                "frames_written": len(frames),
                "total_video_frames": len(frames),
                "processing_time_seconds": processing_time,
                "metrics_json": summary.model_dump_json(),
                "completed_at": completed_at.isoformat(),
            },
            db_path,
        )
        logger.info(
            "Crack mapping job %s finished: %d frames (%d with cracks, "
            "mean width %.3f px) in %.3fs -> %s",
            job_id,
            summary.frames_analyzed,
            summary.frames_with_cracks,
            summary.mean_width_px,
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
        logger.exception("Crack mapping job %s failed", job_id)
