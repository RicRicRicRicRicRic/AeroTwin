"""Defect metrics for crack mapping (Phase 3).

Quantifies detected crack masks per frame — pixel count, area ratio, skeleton
length, and mean width (the classic *width = area / length* relation) — for
the thesis results chapter. Zhang-Suen thinning is implemented in pure NumPy
so no extra dependency is needed, and every computation is deterministic.
Per-frame metrics are persisted to ``aerotwin.db`` (``frame_defect_metrics``).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from ..core.database import get_connection
from ..schemas.processing import CrackFrameMetrics, CrackRunSummary

logger = logging.getLogger("aerotwin.defects")


def zhang_suen_thinning(binary_mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning of a boolean mask → 1-px-wide boolean skeleton.

    Iteratively deletes border pixels under the standard two sub-step
    conditions until convergence. Deterministic; pure NumPy over the whole
    mask per pass (OpenCV's base wheel ships no thinning implementation).
    """
    image = np.asarray(binary_mask, dtype=bool).astype(np.uint8)
    if not image.any():
        return image.astype(bool)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            padded = np.pad(image, 1)
            p2 = padded[0:-2, 1:-1]  # N
            p3 = padded[0:-2, 2:]  # NE
            p4 = padded[1:-1, 2:]  # E
            p5 = padded[2:, 2:]  # SE
            p6 = padded[2:, 1:-1]  # S
            p7 = padded[2:, :-2]  # SW
            p8 = padded[1:-1, :-2]  # W
            p9 = padded[0:-2, :-2]  # NW
            neighbours = (p2, p3, p4, p5, p6, p7, p8, p9)
            neighbour_sum = sum(neighbours)  # B(p1): foreground neighbours
            sequence = neighbours + (p2,)
            transitions = np.zeros_like(image)
            for index in range(8):  # A(p1): 0->1 transitions around p1
                transitions += (
                    (sequence[index] == 0) & (sequence[index + 1] == 1)
                ).astype(np.uint8)
            common = (
                (image == 1)
                & (transitions == 1)
                & (neighbour_sum >= 2)
                & (neighbour_sum <= 6)
            )
            if step == 0:
                delete = common & (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                delete = common & (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            if delete.any():
                image[delete] = 0
                changed = True
    return image.astype(bool)


def compute_crack_metrics(frame_filename: str, crack_mask: np.ndarray) -> CrackFrameMetrics:
    """Quantify one frame's boolean crack mask (pixels, length, width, count)."""
    mask = np.asarray(crack_mask, dtype=bool)
    pixel_count = int(mask.sum())
    total_pixels = int(mask.size)
    if pixel_count == 0:
        return CrackFrameMetrics(
            frame_filename=frame_filename,
            crack_pixel_count=0,
            crack_area_ratio=0.0,
            crack_length_px=0.0,
            mean_width_px=0.0,
            component_count=0,
        )
    skeleton = zhang_suen_thinning(mask)
    length_px = float(skeleton.sum())
    mean_width_px = pixel_count / length_px if length_px > 0 else 0.0
    components = int(cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0]) - 1
    area_ratio = 100.0 * pixel_count / total_pixels
    return CrackFrameMetrics(
        frame_filename=frame_filename,
        crack_pixel_count=pixel_count,
        crack_area_ratio=round(area_ratio, 6),
        crack_length_px=round(length_px, 3),
        mean_width_px=round(mean_width_px, 4),
        component_count=max(components, 0),
    )


def summarize_crack_metrics(rows: Sequence[CrackFrameMetrics]) -> CrackRunSummary:
    """Aggregate per-frame metrics into run-level summary statistics."""
    frames_with_cracks = [row for row in rows if row.crack_pixel_count > 0]
    return CrackRunSummary(
        frames_analyzed=len(rows),
        frames_with_cracks=len(frames_with_cracks),
        total_crack_pixels=sum(row.crack_pixel_count for row in rows),
        mean_area_ratio=round(
            float(np.mean([row.crack_area_ratio for row in rows])), 6
        )
        if rows
        else 0.0,
        mean_crack_length_px=round(
            float(np.mean([row.crack_length_px for row in frames_with_cracks])), 3
        )
        if frames_with_cracks
        else 0.0,
        mean_width_px=round(
            float(np.mean([row.mean_width_px for row in frames_with_cracks])), 4
        )
        if frames_with_cracks
        else 0.0,
        longest_crack_px=round(
            max((row.crack_length_px for row in rows), default=0.0), 3
        ),
    )


def log_frame_metrics(
    job_id: str,
    rows: Sequence[CrackFrameMetrics],
    db_path: Path | None = None,
) -> None:
    """Persist per-frame metrics for a job (batch insert; idempotent per job/frame)."""
    if not rows:
        return
    computed_at = datetime.now(timezone.utc).isoformat()
    payload = [
        (
            job_id,
            row.frame_filename,
            row.crack_pixel_count,
            row.crack_area_ratio,
            row.crack_length_px,
            row.mean_width_px,
            row.component_count,
            computed_at,
        )
        for row in rows
    ]
    with get_connection(db_path) as connection:
        connection.executemany(
            """
            INSERT OR REPLACE INTO frame_defect_metrics (
                job_id, frame_filename, crack_pixel_count, crack_area_ratio,
                crack_length_px, mean_width_px, component_count, computed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            payload,
        )
    logger.info("Logged %d frame metric rows for job %s", len(rows), job_id)


def fetch_job_metrics(job_id: str, db_path: Path | None = None) -> list[CrackFrameMetrics]:
    """Return all persisted per-frame metrics for a job (insertion order)."""
    with get_connection(db_path) as connection:
        rows = connection.execute(
            """
            SELECT frame_filename, crack_pixel_count, crack_area_ratio,
                   crack_length_px, mean_width_px, component_count
            FROM frame_defect_metrics WHERE job_id = ? ORDER BY id
            """,
            (job_id,),
        ).fetchall()
    return [
        CrackFrameMetrics(
            frame_filename=row["frame_filename"],
            crack_pixel_count=row["crack_pixel_count"],
            crack_area_ratio=row["crack_area_ratio"],
            crack_length_px=row["crack_length_px"],
            mean_width_px=row["mean_width_px"],
            component_count=row["component_count"],
        )
        for row in rows
    ]
