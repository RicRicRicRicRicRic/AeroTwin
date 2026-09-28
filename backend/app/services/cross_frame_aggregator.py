"""Cross-frame spatial deduplication & global defect registry (Phase 6).

Frame-level detection over-counts: a single diagonal crack that stays in view
for 15 sampled frames yields ~15 detections, and a column seen from three
angles yields three "columns". Before the seismic engine can reason about
*the building* instead of *the frames*, those observations have to be folded
into global entities:

``Crack CRK-0001 — observed in frames 40-55, 16 observations, max length
124 px, max width 4.1 px, mean merge IoU 0.71``

Algorithm (deterministic, dependency-free — no ``shapely`` needed):

1. **Extract** per-frame observations: crack components (bounding box, pixel
   area, skeleton length, mean width) from ``crack_mask_*.png`` and element
   instances (bounding box, label, confidence) from ``elements.json`` —
   falling back to the class masks ``mask_*.png`` when no JSON sidecar exists.
2. **Match** observations of frame *n* against every entity last seen within
   ``max_frame_gap`` frames using axis-aligned bounding-box IoU.
3. **Assign** greedily best-IoU-first (one observation and one entity per
   frame, so two parallel cracks in the same frame can never collapse into one
   entity); unmatched observations open a new registry entry.
4. **Merge** into union bounding boxes and per-entity extrema (max/mean area,
   length, width, confidence, worst crack density), keeping full per-frame
   provenance for the thesis audit trail.

Outputs: ``data/processed/aggregated/<video>/job_<id8>/aggregated_defects.json``
+ ``manifest.json``, and the ``global_crack_defects`` /
``global_element_instances`` tables in ``aerotwin.db``. Raw inputs are never
modified and unique job directories guarantee outputs are never overwritten.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..core.config import settings
from ..core.database import get_connection, get_job_record, update_job_record
from ..models.crack_detector import JOB_TYPE_CRACK_MAPPING
from ..models.structural_element_detector import JOB_TYPE_ELEMENT_DETECTION
from ..schemas.defect_registry import (
    AggregationRequest,
    AggregationRun,
    AggregationSummary,
    DefectObservation,
    DefectRegistry,
    GlobalCrackEntity,
    GlobalElementEntity,
)
from ..schemas.processing import JobStatus
from .defect_metrics import zhang_suen_thinning

logger = logging.getLogger("aerotwin.aggregator")

#: ``processing_jobs.job_type`` value of an aggregation run.
JOB_TYPE_AGGREGATION: str = "cross_frame_aggregation"
#: Default merge criteria (documented, deterministic).
DEFAULT_IOU_THRESHOLD: float = 0.30
DEFAULT_MAX_FRAME_GAP: int = 10
#: Crack components / element components below this many pixels are noise.
MIN_CRACK_COMPONENT_AREA: int = 8
MIN_ELEMENT_COMPONENT_AREA: int = 16
#: Index → label of the element class mask written by the element detector.
ELEMENT_MASK_CLASSES: tuple[str, ...] = ("background", "column", "beam", "wall")

REGISTRY_FILENAME: str = "aggregated_defects.json"
MANIFEST_FILENAME: str = "manifest.json"

_FRAME_INDEX_RE = re.compile(r"(\d+)")
CrackBox = tuple[int, int, int, int]


def frame_index_from_filename(filename: str) -> int:
    """Trailing numeric run of a frame filename (``frame_000042.jpg`` → 42).

    Frames without any digits collapse to ``0`` so ordering stays defined.
    """
    digits = _FRAME_INDEX_RE.findall(Path(filename).stem)
    if not digits:
        return 0
    try:
        return int(digits[-1])
    except ValueError:  # pragma: no cover - absurdly long digit run
        return 0


def bbox_iou(first: CrackBox, second: CrackBox) -> float:
    """Intersection-over-union of two ``(x, y, width, height)`` boxes.

    Returns ``0.0`` for degenerate boxes so callers can treat "no overlap" and
    "invalid geometry" identically.
    """
    ax1, ay1, aw, ah = first
    bx1, by1, bw, bh = second
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    inter_w = min(ax2, bx2) - max(ax1, bx1)
    inter_h = min(ay2, by2) - max(ay1, by1)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    intersection = float(inter_w * inter_h)
    union = float(aw * ah) + float(bw * bh) - intersection
    return intersection / union if union > 0 else 0.0


@dataclass(frozen=True)
class FrameObservations:
    """All detections of one frame, in a stable (top-left first) order."""

    frame_filename: str
    frame_index: int
    frame_id: str
    observations: tuple[DefectObservation, ...]


# ---------------------------------------------------------------------------
# Frame → observation extraction (crack components & element instances)
# ---------------------------------------------------------------------------
class CrackMaskSource:
    """Lazily decoded, cached binary crack masks keyed by frame id."""

    def __init__(self, crack_output_dir: Path | None) -> None:
        self.directory = crack_output_dir
        self._cache: dict[str, np.ndarray | None] = {}

    def get(self, frame_id: str) -> np.ndarray | None:
        """Binary mask of ``crack_mask_<frame_id>.png`` (``None`` when absent)."""
        if self.directory is None:
            return None
        if frame_id in self._cache:
            return self._cache[frame_id]
        path = self.directory / f"crack_mask_{frame_id}.png"
        mask: np.ndarray | None = None
        if path.is_file():
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                logger.warning("Unreadable crack mask: %s", path)
            else:
                mask = image > 0
        self._cache[frame_id] = mask
        return mask

    @property
    def frame_ids(self) -> list[str]:
        """Frame ids that actually have a crack mask on disk (sorted)."""
        if self.directory is None or not self.directory.is_dir():
            return []
        return sorted(p.stem.removeprefix("crack_mask_") for p in self.directory.glob("crack_mask_*.png"))


def load_crack_frame_names(crack_output_dir: Path) -> dict[str, str]:
    """Map ``frame_id`` → original frame filename using the crack job's ``metrics.json``.

    Keeps registry provenance aligned with ``frame_defect_metrics.frame_filename``
    so frames can be joined across tables. An absent or unreadable sidecar
    simply yields an empty mapping (frame ids are used as fallback names).
    """
    path = crack_output_dir / "metrics.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Ignoring unreadable crack metrics sidecar %s: %s", path, exc)
        return {}
    mapping: dict[str, str] = {}
    frames = payload.get("frames") if isinstance(payload, dict) else None
    for entry in frames or []:
        filename = str(entry.get("frame_filename", "")) if isinstance(entry, dict) else ""
        if not filename:
            continue
        mapping[Path(filename).stem.removeprefix("frame_")] = filename
    return mapping


def _observation_sort_key(observation: DefectObservation) -> tuple[int, int, int]:
    """Deterministic in-frame order: top-left-most first, then by size."""
    return (observation.y, observation.x, -(observation.width * observation.height))


def crack_observations_for_mask(
    mask: np.ndarray,
    *,
    frame_filename: str,
    frame_index: int,
) -> list[DefectObservation]:
    """Turn one binary crack mask into per-component observations.

    Each connected component contributes its bounding box, pixel area, Zhang-Suen
    skeleton length, and mean stroke width (``area / length``), matching the
    definitions used by :mod:`app.services.defect_metrics`.
    """
    binary = np.asarray(mask) > 0
    if not binary.any():
        return []
    skeleton = zhang_suen_thinning(binary)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    observations: list[DefectObservation] = []
    for component in range(1, count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        if area < MIN_CRACK_COMPONENT_AREA:
            continue
        x = int(stats[component, cv2.CC_STAT_LEFT])
        y = int(stats[component, cv2.CC_STAT_TOP])
        width = max(int(stats[component, cv2.CC_STAT_WIDTH]), 1)
        height = max(int(stats[component, cv2.CC_STAT_HEIGHT]), 1)
        length = float((skeleton & (labels == component)).sum())
        if length <= 0.0:
            length = float(max(width, height))
        observations.append(
            DefectObservation(
                frame_filename=frame_filename,
                frame_index=frame_index,
                x=x,
                y=y,
                width=width,
                height=height,
                pixel_count=area,
                length_px=round(length, 3),
                width_px=round(area / length, 4),
            )
        )
    observations.sort(key=_observation_sort_key)
    return observations


def extract_crack_frames(crack_output_dir: Path) -> list[FrameObservations]:
    """Read every ``crack_mask_*.png`` of a crack-mapping job into observations.

    Raises:
        FileNotFoundError: *crack_output_dir* does not exist.
    """
    if not crack_output_dir.is_dir():
        raise FileNotFoundError(f"Crack map output directory not found: {crack_output_dir}")
    frame_names = load_crack_frame_names(crack_output_dir)
    source = CrackMaskSource(crack_output_dir)
    frames: list[FrameObservations] = []
    for frame_id in source.frame_ids:
        mask = source.get(frame_id)
        if mask is None:
            continue
        frame_filename = frame_names.get(frame_id, f"frame_{frame_id}")
        observations = crack_observations_for_mask(
            mask,
            frame_filename=frame_filename,
            frame_index=frame_index_from_filename(frame_filename),
        )
        frames.append(
            FrameObservations(
                frame_filename=frame_filename,
                frame_index=frame_index_from_filename(frame_filename),
                frame_id=frame_id,
                observations=tuple(observations),
            )
        )
    return frames


def _density_in_box(mask: np.ndarray, observation: DefectObservation) -> float:
    """Crack-pixel density inside one observation's bounding box (0-1)."""
    return _density_in_box_geometry(
        mask, observation.x, observation.y, observation.width, observation.height
    )


def _density_in_box_geometry(
    mask: np.ndarray, x: int, y: int, width: int, height: int
) -> float:
    """Crack-pixel density inside a raw ``(x, y, width, height)`` box, clipped."""
    mask_height, mask_width = mask.shape[:2]
    x2 = min(x + width, mask_width)
    y2 = min(y + height, mask_height)
    crop = mask[max(y, 0) : y2, max(x, 0) : x2]
    return float(crop.mean()) if crop.size else 0.0


def _element_frames_from_json(
    payload: dict[str, Any],
    cracks: CrackMaskSource,
) -> list[FrameObservations]:
    """Parse an element detector ``elements.json`` into per-frame observations."""
    frames: list[FrameObservations] = []
    for entry in payload.get("frames") or []:
        if not isinstance(entry, dict):
            continue
        frame_filename = str(entry.get("frame", entry.get("frame_filename", "")))
        if not frame_filename:
            continue
        frame_id = Path(frame_filename).stem.removeprefix("frame_")
        frame_index = frame_index_from_filename(frame_filename)
        mask = cracks.get(frame_id)
        observations: list[DefectObservation] = []
        for element in entry.get("elements") or []:
            if not isinstance(element, dict):
                continue
            width = int(element.get("width", 0))
            height = int(element.get("height", 0))
            if width <= 0 or height <= 0:
                continue
            confidence = element.get("confidence")
            box = (int(element.get("x", 0)), int(element.get("y", 0)), width, height)
            observations.append(
                DefectObservation(
                    frame_filename=frame_filename,
                    frame_index=frame_index,
                    x=box[0],
                    y=box[1],
                    width=width,
                    height=height,
                    pixel_count=width * height,
                    label=str(element.get("label", "unknown")),
                    confidence=float(confidence) if confidence is not None else None,
                    crack_density=(
                        round(_density_in_box_geometry(mask, *box), 6)
                        if mask is not None
                        else None
                    ),
                )
            )
        observations.sort(key=_observation_sort_key)
        frames.append(
            FrameObservations(
                frame_filename=frame_filename,
                frame_index=frame_index,
                frame_id=frame_id,
                observations=tuple(observations),
            )
        )
    return frames


def _element_frames_from_masks(
    element_output_dir: Path,
    cracks: CrackMaskSource,
) -> list[FrameObservations]:
    """Fallback extraction: per-class connected components of ``mask_*.png``.

    Used when a job directory has element class masks but no ``elements.json``
    (hand-written fixtures, or a run that crashed before writing the sidecar).
    """
    frames: list[FrameObservations] = []
    for mask_path in sorted(element_output_dir.glob("mask_*.png")):
        frame_id = mask_path.stem.removeprefix("mask_")
        class_mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if class_mask is None:
            logger.warning("Unreadable element mask: %s", mask_path)
            continue
        if class_mask.ndim != 2:
            logger.warning("Ignoring non-indexed element mask: %s", mask_path)
            continue
        frame_filename = f"frame_{frame_id}"
        frame_index = frame_index_from_filename(frame_filename)
        crack_mask = cracks.get(frame_id)
        observations: list[DefectObservation] = []
        for class_index in (int(value) for value in np.unique(class_mask) if value > 0):
            label = (
                ELEMENT_MASK_CLASSES[class_index]
                if class_index < len(ELEMENT_MASK_CLASSES)
                else f"class_{class_index}"
            )
            binary = (class_mask == class_index).astype(np.uint8)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            for component in range(1, count):
                area = int(stats[component, cv2.CC_STAT_AREA])
                if area < MIN_ELEMENT_COMPONENT_AREA:
                    continue
                component_mask = labels == component
                density = (
                    round(float(crack_mask[component_mask].mean()), 6)
                    if crack_mask is not None
                    else None
                )
                observations.append(
                    DefectObservation(
                        frame_filename=frame_filename,
                        frame_index=frame_index,
                        x=int(stats[component, cv2.CC_STAT_LEFT]),
                        y=int(stats[component, cv2.CC_STAT_TOP]),
                        width=max(int(stats[component, cv2.CC_STAT_WIDTH]), 1),
                        height=max(int(stats[component, cv2.CC_STAT_HEIGHT]), 1),
                        pixel_count=area,
                        label=label,
                        confidence=1.0,
                        crack_density=density,
                    )
                )
        observations.sort(key=_observation_sort_key)
        frames.append(
            FrameObservations(
                frame_filename=frame_filename,
                frame_index=frame_index,
                frame_id=frame_id,
                observations=tuple(observations),
            )
        )
    return frames


def extract_element_frames(
    element_output_dir: Path,
    cracks: CrackMaskSource | None = None,
) -> list[FrameObservations]:
    """Read an element-detection job's instances into per-frame observations.

    Prefers the ``elements.json`` sidecar (label + confidence per box) and
    falls back to the class masks. When *cracks* is given, each observation also
    records the crack-pixel density inside it.

    Raises:
        FileNotFoundError: *element_output_dir* does not exist.
    """
    if not element_output_dir.is_dir():
        raise FileNotFoundError(
            f"Element detection output directory not found: {element_output_dir}"
        )
    mask_source = cracks if cracks is not None else CrackMaskSource(None)
    sidecar = element_output_dir / "elements.json"
    if sidecar.is_file():
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("Ignoring unreadable element sidecar %s: %s", sidecar, exc)
        else:
            if isinstance(payload, dict):
                return _element_frames_from_json(payload, mask_source)
    return _element_frames_from_masks(element_output_dir, mask_source)


# ---------------------------------------------------------------------------
# Deduplication: frame-local observations → global entities
# ---------------------------------------------------------------------------
@dataclass
class EntityAccumulator:
    """Mutable cross-frame state of one global entity (union box + provenance)."""

    label: str
    observations: list[DefectObservation] = field(default_factory=list)
    match_ious: list[float] = field(default_factory=list)
    left: int = 0
    top: int = 0
    right: int = 0
    bottom: int = 0
    last_frame_index: int = -1

    def add(self, observation: DefectObservation, iou: float | None) -> None:
        """Attach an observation, growing the union bounding box."""
        if not self.observations:
            self.left = observation.x
            self.top = observation.y
            self.right = observation.x + observation.width
            self.bottom = observation.y + observation.height
        else:
            self.left = min(self.left, observation.x)
            self.top = min(self.top, observation.y)
            self.right = max(self.right, observation.x + observation.width)
            self.bottom = max(self.bottom, observation.y + observation.height)
        if iou is not None:
            self.match_ious.append(iou)
        self.observations.append(observation)
        self.last_frame_index = max(self.last_frame_index, observation.frame_index)

    @property
    def box(self) -> CrackBox:
        """Union bounding box of every observation merged so far."""
        return (self.left, self.top, max(self.right - self.left, 1), max(self.bottom - self.top, 1))

    @property
    def first(self) -> DefectObservation:
        """Earliest observation (frames are consumed in frame order)."""
        return self.observations[0]

    @property
    def last(self) -> DefectObservation:
        """Most recent observation."""
        return self.observations[-1]

    @property
    def mean_match_iou(self) -> float | None:
        """Mean merge IoU, or ``None`` for a single-observation entity."""
        if not self.match_ious:
            return None
        return round(sum(self.match_ious) / len(self.match_ious), 6)


def _observation_box(observation: DefectObservation) -> CrackBox:
    return (observation.x, observation.y, observation.width, observation.height)


def deduplicate_observations(
    frames: list[FrameObservations],
    *,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    max_frame_gap: int = DEFAULT_MAX_FRAME_GAP,
    match_label: bool = True,
) -> list[EntityAccumulator]:
    """Greedy best-IoU-first temporal linking of frame-local observations.

    Frames are consumed in ascending frame order. Every observation of the
    current frame is matched against entities last seen within *max_frame_gap*
    frames; candidate pairs are assigned highest IoU first, with at most one
    observation and one entity per frame, so two distinct defects overlapping in
    space can never be folded into a single entity. Unmatched observations open
    new entities. The result is deterministic for a given input ordering.

    Similarity is measured between the candidate and the entity's *last*
    observation rather than its running union box: the union grows with every
    merged frame, which would make a steadily drifting crack overlap its own
    history less and less until the track silently splits.
    """
    entities: list[EntityAccumulator] = []
    for frame in sorted(frames, key=lambda item: (item.frame_index, item.frame_filename)):
        candidates: list[tuple[float, int, int, float]] = []
        for entity_index, entity in enumerate(entities):
            if frame.frame_index - entity.last_frame_index > max_frame_gap:
                continue
            for observation_index, observation in enumerate(frame.observations):
                if match_label and _label_of(observation) != entity.label:
                    continue
                iou = bbox_iou(_observation_box(entity.last), _observation_box(observation))
                if iou >= iou_threshold:
                    candidates.append((-iou, entity_index, observation_index, iou))
        candidates.sort()
        used_entities: set[int] = set()
        used_observations: set[int] = set()
        for _, entity_index, observation_index, iou in candidates:
            if entity_index in used_entities or observation_index in used_observations:
                continue
            used_entities.add(entity_index)
            used_observations.add(observation_index)
            entities[entity_index].add(frame.observations[observation_index], iou)
        for observation_index, observation in enumerate(frame.observations):
            if observation_index in used_observations:
                continue
            fresh = EntityAccumulator(label=_label_of(observation))
            fresh.add(observation, None)
            entities.append(fresh)
    return entities


def _label_of(observation: DefectObservation) -> str:
    """Entity label of an observation: its element class, or ``crack``."""
    return observation.label or "crack"


# ---------------------------------------------------------------------------
# Accumulator → persisted registry entities
# ---------------------------------------------------------------------------
def _mean(values: list[float]) -> float:
    """Arithmetic mean of a non-empty list."""
    return sum(values) / len(values)


def _entities_in_order(entities: list[EntityAccumulator]) -> list[EntityAccumulator]:
    """Stable ordering: by first-seen frame, then top-left of the union box."""
    return sorted(
        entities,
        key=lambda entity: (entity.first.frame_index, entity.top, entity.left, entity.label),
    )


def _provenance(
    entity: EntityAccumulator, include_observations: bool
) -> list[dict[str, Any]]:
    """Per-frame observation records, annotated with their merge IoU."""
    records: list[dict[str, Any]] = []
    for position, observation in enumerate(entity.observations):
        if position > 0:
            observation = observation.model_copy(update={"match_iou": entity.match_ious[position - 1]})
        record = observation.model_dump(mode="json")
        if not include_observations:
            record.pop("match_iou", None)
        records.append(record)
    return records


def _observations(
    entity: EntityAccumulator, include_observations: bool
) -> list[DefectObservation]:
    """Observations with merge IoU attached, ready for the API response."""
    annotated: list[DefectObservation] = []
    for position, observation in enumerate(entity.observations):
        if position > 0:
            observation = observation.model_copy(update={"match_iou": entity.match_ious[position - 1]})
        annotated.append(observation)
    if include_observations:
        return annotated
    strongest = _strongest_observation(entity)
    return [strongest]


def _strongest_observation(entity: EntityAccumulator) -> DefectObservation:
    """Most informative observation: longest crack / largest area, ties → latest."""
    return max(
        entity.observations,
        key=lambda item: (item.length_px, item.pixel_count, item.frame_index),
    )


def build_crack_entities(
    entities: list[EntityAccumulator],
    *,
    include_observations: bool = True,
) -> list[GlobalCrackEntity]:
    """Convert crack accumulators into registry entities with extrema + provenance."""
    built: list[GlobalCrackEntity] = []
    for number, entity in enumerate(_entities_in_order(entities), start=1):
        observations = _observations(entity, include_observations)
        areas = [float(item.pixel_count) for item in entity.observations]
        lengths = [item.length_px for item in entity.observations]
        widths = [item.width_px for item in entity.observations]
        first, last = entity.first, entity.last
        built.append(
            GlobalCrackEntity(
                defect_id=f"CRK-{number:04d}",
                observation_count=len(entity.observations),
                first_frame_filename=first.frame_filename,
                last_frame_filename=last.frame_filename,
                first_frame_index=first.frame_index,
                last_frame_index=last.frame_index,
                union_x=entity.left,
                union_y=entity.top,
                union_width=entity.box[2],
                union_height=entity.box[3],
                pixel_count_max=int(max(areas)),
                pixel_count_mean=round(_mean(areas), 3),
                length_px_max=round(max(lengths), 3),
                length_px_mean=round(_mean(lengths), 3),
                width_px_max=round(max(widths), 4),
                width_px_mean=round(_mean(widths), 4),
                mean_match_iou=entity.mean_match_iou,
                representative=_strongest_observation(entity),
                observations=observations,
            )
        )
    return built


def build_element_entities(
    entities: list[EntityAccumulator],
    *,
    include_observations: bool = True,
) -> list[GlobalElementEntity]:
    """Convert element accumulators into registry entities with extrema."""
    built: list[GlobalElementEntity] = []
    counts: dict[str, int] = {}
    for entity in _entities_in_order(entities):
        counts[entity.label] = counts.get(entity.label, 0) + 1
        prefix = f"ELM-{entity.label.upper()}"
        confidences = [
            item.confidence for item in entity.observations if item.confidence is not None
        ]
        densities = [
            item.crack_density
            for item in entity.observations
            if item.crack_density is not None
        ]
        first, last = entity.first, entity.last
        strongest = _strongest_observation(entity)
        built.append(
            GlobalElementEntity(
                defect_id=f"{prefix}-{counts[entity.label]:04d}",
                label=entity.label,
                observation_count=len(entity.observations),
                first_frame_filename=first.frame_filename,
                last_frame_filename=last.frame_filename,
                first_frame_index=first.frame_index,
                last_frame_index=last.frame_index,
                union_x=entity.left,
                union_y=entity.top,
                union_width=entity.box[2],
                union_height=entity.box[3],
                confidence_max=round(max(confidences), 4) if confidences else None,
                confidence_mean=round(_mean(confidences), 4) if confidences else None,
                crack_density_max=round(max(densities), 6) if densities else None,
                mean_match_iou=entity.mean_match_iou,
                representative=strongest,
                observations=_observations(entity, include_observations),
            )
        )
    return built


# ---------------------------------------------------------------------------
# Summary & persistence
# ---------------------------------------------------------------------------
def summarize_aggregation(
    crack_frames: list[FrameObservations],
    crack_entities: list[EntityAccumulator],
    element_frames: list[FrameObservations] | None,
    element_entities: list[EntityAccumulator] | None,
    *,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    max_frame_gap: int = DEFAULT_MAX_FRAME_GAP,
) -> AggregationSummary:
    """Dedup statistics quoted in the report, the DB row, and the API response."""
    raw_cracks = sum(len(frame.observations) for frame in crack_frames)
    unique_cracks = len(crack_entities)
    raw_elements = (
        sum(len(frame.observations) for frame in element_frames)
        if element_frames is not None
        else 0
    )
    elements_by_label: dict[str, int] | None = None
    unique_elements: int | None = None
    if element_entities is not None:
        unique_elements = len(element_entities)
        elements_by_label = {}
        for entity in element_entities:
            elements_by_label[entity.label] = elements_by_label.get(entity.label, 0) + 1
    return AggregationSummary(
        frames_analyzed=sum(1 for item in crack_frames if item.observations),
        frames_with_cracks=sum(1 for item in crack_frames if item.observations),
        raw_crack_observations=raw_cracks,
        unique_crack_defects=unique_cracks,
        raw_element_observations=raw_elements,
        unique_element_instances=unique_elements,
        mean_observations_per_crack=(
            round(raw_cracks / unique_cracks, 4) if unique_cracks else 0.0
        ),
        crack_dedup_ratio=(
            round(1.0 - unique_cracks / raw_cracks, 6) if raw_cracks else 0.0
        ),
        crack_dedup_reduction=max(raw_cracks - unique_cracks, 0),
        elements_by_label=elements_by_label,
        iou_threshold=iou_threshold,
        max_frame_gap=max_frame_gap,
    )


def _provenance_json(entity: EntityAccumulator) -> str:
    """Serialised per-frame provenance stored in the registry tables."""
    return json.dumps(_provenance(entity, True))


def persist_registry(
    job_id: str,
    registry: DefectRegistry,
    crack_entities: list[EntityAccumulator],
    element_entities: list[EntityAccumulator],
    *,
    db_path: Path | None = None,
) -> None:
    """Replace any rows of a previous attempt with this run's global entities.

    Re-running an aggregation job is idempotent: rows are deleted by ``job_id``
    before insertion, so a retried job never duplicates its registry.
    """
    computed_at = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as connection:
        connection.execute("DELETE FROM global_crack_defects WHERE job_id = ?", (job_id,))
        connection.execute(
            "DELETE FROM global_element_instances WHERE job_id = ?", (job_id,)
        )
        for entity, source in zip(crack_entities, registry.cracks, strict=True):
            connection.execute(
                """
                INSERT INTO global_crack_defects (
                    id, job_id, source_crack_job_id, label, observation_count,
                    first_frame_index, last_frame_index, first_frame_filename,
                    last_frame_filename, union_x, union_y, union_width, union_height,
                    pixel_count_max, pixel_count_mean, length_px_max, length_px_mean,
                    width_px_max, width_px_mean, mean_match_iou, provenance_json,
                    computed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"{job_id}:{source.defect_id}",
                    job_id,
                    registry.crack_job_id,
                    source.label,
                    source.observation_count,
                    source.first_frame_index,
                    source.last_frame_index,
                    source.first_frame_filename,
                    source.last_frame_filename,
                    source.union_x,
                    source.union_y,
                    source.union_width,
                    source.union_height,
                    source.pixel_count_max,
                    source.pixel_count_mean,
                    source.length_px_max,
                    source.length_px_mean,
                    source.width_px_max,
                    source.width_px_mean,
                    source.mean_match_iou,
                    _provenance_json(entity),
                    computed_at,
                ),
            )
        for entity, source in zip(element_entities, registry.elements, strict=True):
            connection.execute(
                """
                INSERT INTO global_element_instances (
                    id, job_id, source_crack_job_id, source_element_job_id, label,
                    observation_count, first_frame_index, last_frame_index,
                    first_frame_filename, last_frame_filename, union_x, union_y,
                    union_width, union_height, confidence_max, confidence_mean,
                    crack_density_max, mean_match_iou, provenance_json, computed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"{job_id}:{source.defect_id}",
                    job_id,
                    registry.crack_job_id,
                    registry.element_job_id or "",
                    source.label,
                    source.observation_count,
                    source.first_frame_index,
                    source.last_frame_index,
                    source.first_frame_filename,
                    source.last_frame_filename,
                    source.union_x,
                    source.union_y,
                    source.union_width,
                    source.union_height,
                    source.confidence_max,
                    source.confidence_mean,
                    source.crack_density_max,
                    source.mean_match_iou,
                    _provenance_json(entity),
                    computed_at,
                ),
            )


# ---------------------------------------------------------------------------
# Background job
# ---------------------------------------------------------------------------
def _completed_job_output_dir(
    job_id: str, expected_type: str, db_path: Path | None
) -> Path:
    """Validate a source job is completed and return its output directory.

    Raises:
        ValueError: The job is unknown, the wrong type, not completed, or has no
            recorded output directory.
    """
    row = get_job_record(job_id, db_path)
    if row is None:
        raise ValueError(f"Unknown {expected_type} job: {job_id}")
    if str(row["job_type"]) != expected_type:
        raise ValueError(
            f"Job {job_id} is a {row['job_type']} run, not a {expected_type} run"
        )
    if str(row["status"]) != JobStatus.COMPLETED.value:
        raise ValueError(
            f"Job {job_id} is {row['status']}, not completed — "
            "aggregation needs finished inputs"
        )
    if not row["output_dir"]:
        raise ValueError(f"Job {job_id} completed without an output directory")
    return Path(str(row["output_dir"]))


def aggregate_results(
    crack_job_id: str,
    params: AggregationRequest,
    *,
    db_path: Path | None = None,
) -> DefectRegistry:
    """Run the whole aggregation in-process and return the built registry.

    Used by tests and by any caller that wants the result synchronously; the
    API layer uses :func:`run_aggregation_job` instead.

    Raises:
        ValueError: A source job is unknown, not completed, or has no crack masks.
    """
    crack_dir = _completed_job_output_dir(crack_job_id, JOB_TYPE_CRACK_MAPPING, db_path)
    element_dir = (
        _completed_job_output_dir(
            params.element_job_id, JOB_TYPE_ELEMENT_DETECTION, db_path
        )
        if params.element_job_id
        else None
    )
    crack_frames = extract_crack_frames(crack_dir)
    if not crack_frames:
        raise ValueError(f"No crack masks found in {crack_dir.name} — nothing to aggregate")
    crack_accumulators = deduplicate_observations(
        crack_frames,
        iou_threshold=params.iou_threshold,
        max_frame_gap=params.max_frame_gap,
    )
    element_frames: list[FrameObservations] | None = None
    element_accumulators: list[EntityAccumulator] | None = None
    if element_dir is not None:
        element_frames = extract_element_frames(element_dir, CrackMaskSource(crack_dir))
        element_accumulators = deduplicate_observations(
            element_frames,
            iou_threshold=params.iou_threshold,
            max_frame_gap=params.max_frame_gap,
        )
    return DefectRegistry(
        aggregation_job_id="",
        crack_job_id=crack_job_id,
        element_job_id=params.element_job_id,
        video_filename=crack_dir.parent.name,
        generated_at=datetime.now(timezone.utc),
        params=params.model_dump(mode="json"),
        summary=summarize_aggregation(
            crack_frames,
            crack_accumulators,
            element_frames,
            element_accumulators,
            iou_threshold=params.iou_threshold,
            max_frame_gap=params.max_frame_gap,
        ),
        cracks=build_crack_entities(
            crack_accumulators, include_observations=params.include_observations
        ),
        elements=(
            build_element_entities(
                element_accumulators, include_observations=params.include_observations
            )
            if element_accumulators is not None
            else []
        ),
    )

# ---------------------------------------------------------------------------
# Read side: rebuild an ``AggregationRun`` for the API
# ---------------------------------------------------------------------------
def load_aggregation_run(
    job_id: str,
    *,
    db_path: Path | None = None,
) -> AggregationRun | None:
    """Rebuild a completed/failed aggregation run response from the database.

    Returns ``None`` when the job does not exist or is not an aggregation run.
    The registry sidecar supplies cracks/elements; a missing or unreadable file
    yields an empty registry rather than an error (the job row is the source of
    truth for status and metrics).
    """
    row = get_job_record(job_id, db_path)
    if row is None or str(row["job_type"]) != JOB_TYPE_AGGREGATION:
        return None
    try:
        params = AggregationRequest.model_validate(
            json.loads(row["params_json"]) if row["params_json"] else {}
        )
    except (OSError, ValueError):
        params = AggregationRequest(crack_job_id="")
    summary: AggregationSummary | None = None
    if row["metrics_json"]:
        try:
            summary = AggregationSummary.model_validate_json(row["metrics_json"])
        except ValueError:
            logger.warning("Ignoring unreadable aggregation summary for %s", job_id)
    cracks: list[GlobalCrackEntity] = []
    elements: list[GlobalElementEntity] = []
    output_dir = str(row["output_dir"]) if row["output_dir"] else None
    if output_dir and row["status"] == JobStatus.COMPLETED.value:
        registry_path = Path(output_dir) / REGISTRY_FILENAME
        if registry_path.is_file():
            try:
                registry = DefectRegistry.model_validate_json(
                    registry_path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError) as exc:
                logger.warning("Ignoring unreadable registry %s: %s", registry_path, exc)
            else:
                cracks = registry.cracks
                elements = registry.elements
    return AggregationRun(
        job_id=job_id,
        status=JobStatus(str(row["status"])),
        video_filename=str(row["video_filename"]),
        crack_job_id=params.crack_job_id or None,
        element_job_id=params.element_job_id,
        params=params.model_dump(mode="json"),
        summary=summary,
        cracks=cracks,
        elements=elements,
        output_dir=output_dir,
        error=str(row["error"]) if row["error"] else None,
        processing_time_seconds=row["processing_time_seconds"],
        created_at=datetime.fromisoformat(str(row["created_at"])),
        completed_at=(
            datetime.fromisoformat(str(row["completed_at"])) if row["completed_at"] else None
        ),
    )


def run_aggregation_job(
    job_id: str,
    crack_job_id: str,
    params: AggregationRequest,
    *,
    output_root: Path | None = None,
    db_path: Path | None = None,
) -> None:
    """Background task: dedup frame-local detections into a global registry.

    Never raises — every failure lands on the job row, and a partially written
    output directory is removed so incomplete runs can never be mistaken for
    results. The job id becomes the registry's ``aggregation_job_id``.
    """
    update_job_record(job_id, {"status": JobStatus.RUNNING.value}, db_path)
    started_at = datetime.now(timezone.utc)
    timer = time.perf_counter()
    root = output_root if output_root is not None else settings.aggregated_dir
    out_dir = root / f"job_{job_id[:8]}"
    try:
        crack_dir = _completed_job_output_dir(crack_job_id, JOB_TYPE_CRACK_MAPPING, db_path)
        video_filename = crack_dir.parent.name
        out_dir = root / Path(video_filename).stem / f"job_{job_id[:8]}"
        out_dir.mkdir(parents=True, exist_ok=False)

        crack_frames = extract_crack_frames(crack_dir)
        if not crack_frames:
            raise ValueError(
                f"No crack masks found in {crack_dir.name} — nothing to aggregate"
            )
        crack_accumulators = deduplicate_observations(
            crack_frames,
            iou_threshold=params.iou_threshold,
            max_frame_gap=params.max_frame_gap,
        )
        element_dir: Path | None = None
        element_frames: list[FrameObservations] | None = None
        element_accumulators: list[EntityAccumulator] | None = None
        if params.element_job_id:
            element_dir = _completed_job_output_dir(
                params.element_job_id, JOB_TYPE_ELEMENT_DETECTION, db_path
            )
            element_frames = extract_element_frames(element_dir, CrackMaskSource(crack_dir))
            element_accumulators = deduplicate_observations(
                element_frames,
                iou_threshold=params.iou_threshold,
                max_frame_gap=params.max_frame_gap,
            )
        summary = summarize_aggregation(
            crack_frames,
            crack_accumulators,
            element_frames,
            element_accumulators,
            iou_threshold=params.iou_threshold,
            max_frame_gap=params.max_frame_gap,
        )
        completed_at = datetime.now(timezone.utc)
        processing_time = time.perf_counter() - timer
        registry = DefectRegistry(
            aggregation_job_id=job_id,
            crack_job_id=crack_job_id,
            element_job_id=params.element_job_id,
            video_filename=video_filename,
            generated_at=completed_at,
            params=params.model_dump(mode="json"),
            summary=summary,
            cracks=build_crack_entities(
                crack_accumulators, include_observations=params.include_observations
            ),
            elements=(
                build_element_entities(
                    element_accumulators, include_observations=params.include_observations
                )
                if element_accumulators is not None
                else []
            ),
        )
        (out_dir / REGISTRY_FILENAME).write_text(
            registry.model_dump_json(indent=2), encoding="utf-8"
        )


        manifest = {
            "job_id": job_id,
            "job_type": JOB_TYPE_AGGREGATION,
            "video_filename": video_filename,
            "crack_job_id": crack_job_id,
            "crack_output_dir": str(crack_dir),
            "element_job_id": params.element_job_id,
            "element_output_dir": str(element_dir) if element_dir else None,
            "frames_processed": len(crack_frames),
            "params": params.model_dump(mode="json"),
            "method": summary.method,
            "unique_crack_defects": summary.unique_crack_defects,
            "unique_element_instances": summary.unique_element_instances,
            "opencv_version": cv2.__version__,
            "numpy_version": np.__version__,
            "processing_time_seconds": processing_time,
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
        }
        (out_dir / MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        persist_registry(
            job_id,
            registry,
            crack_accumulators,
            element_accumulators or [],
            db_path=db_path,
        )
        update_job_record(
            job_id,
            {
                "status": JobStatus.COMPLETED.value,
                "output_dir": str(out_dir),
                "frames_written": len(crack_frames),
                "total_video_frames": len(crack_frames),
                "processing_time_seconds": processing_time,
                "metrics_json": summary.model_dump_json(),
                "completed_at": completed_at.isoformat(),
            },
            db_path,
        )
        logger.info(
            "Aggregation job %s: %d raw crack observations -> %d global defects "
            "(%.1f%% reduction) in %.3fs -> %s",
            job_id,
            summary.raw_crack_observations,
            summary.unique_crack_defects,
            summary.crack_dedup_ratio * 100.0,
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
        logger.exception("Aggregation job %s failed", job_id)


