"""Unit and integration tests for Phase 6 cross-frame aggregation.

Covers the geometry helpers (frame-index parsing, bounding-box IoU), the greedy
best-IoU-first deduplication (including the one-entity-per-frame guard and frame
gaps), mask/JSON observation extraction, entity building with provenance, the
database registry tables, the background job, and the two API routes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.database import (
    create_job_record,
    get_connection,
    init_database,
    update_job_record,
)
from app.main import app
from app.schemas.defect_registry import AggregationRequest, DefectObservation, DefectRegistry
from app.schemas.processing import JobStatus
from app.services.cross_frame_aggregator import (
    JOB_TYPE_AGGREGATION,
    EntityAccumulator,
    FrameObservations,
    bbox_iou,
    build_crack_entities,
    build_element_entities,
    deduplicate_observations,
    extract_crack_frames,
    extract_element_frames,
    frame_index_from_filename,
    load_aggregation_run,
    run_aggregation_job,
)

MASK_SHAPE: tuple[int, int] = (40, 80)


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return init_database(tmp_path / "aerotwin.db")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def observation(
    frame_index: int,
    x: int = 0,
    y: int = 0,
    width: int = 20,
    height: int = 20,
    **extra: object,
) -> DefectObservation:
    return DefectObservation(
        frame_filename=f"frame_{frame_index:06d}.jpg",
        frame_index=frame_index,
        x=x,
        y=y,
        width=width,
        height=height,
        **extra,
    )


def write_crack_job(
    db_path: Path | None,
    job_id: str,
    masks: list[np.ndarray],
    *,
    video_filename: str = "phase6_video",
) -> Path:
    """Register a completed crack-mapping job and write its masks/metrics sidecar."""
    out_dir = settings.crack_maps_dir / video_filename / f"job_{job_id[:8]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for index, mask in enumerate(masks):
        frame_id = f"{index:06d}"
        written = cv2.imwrite(str(out_dir / f"crack_mask_{frame_id}.png"), mask.astype(np.uint8) * 255)
        assert written
        names.append(f"frame_{frame_id}.jpg")
    (out_dir / "metrics.json").write_text(
        json.dumps({"summary": {}, "frames": [{"frame_filename": name} for name in names]}),
        encoding="utf-8",
    )
    create_job_record(
        job_type="crack_mapping",
        video_filename=video_filename,
        params_json=json.dumps({"frames_path": f"{video_filename}/x"}),
        job_id=job_id,
        db_path=db_path,
    )
    update_job_record(
        job_id,
        {"status": JobStatus.COMPLETED.value, "output_dir": str(out_dir)},
        db_path,
    )
    return out_dir


def crack_bar(x: int, y: int, width: int = 12) -> np.ndarray:
    """A single horizontal 3-px-tall crack component inside ``MASK_SHAPE``."""
    mask = np.zeros(MASK_SHAPE, dtype=bool)
    mask[y : y + 3, x : x + width] = True
    return mask


def register_aggregation_job(
    db_path: Path, job_id: str, params: AggregationRequest
) -> None:
    """Create a pending aggregation job row, as the API route would."""
    create_job_record(
        job_type=JOB_TYPE_AGGREGATION,
        video_filename="phase6_video",
        params_json=params.model_dump_json(),
        job_id=job_id,
        db_path=db_path,
    )



def frame(frame_index: int, *observations: DefectObservation) -> FrameObservations:
    return FrameObservations(
        frame_filename=f"frame_{frame_index:06d}.jpg",
        frame_index=frame_index,
        frame_id=f"{frame_index:06d}",
        observations=observations,
    )



# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def test_frame_index_from_filename_parses_trailing_digits() -> None:
    assert frame_index_from_filename("frame_000042.jpg") == 42
    assert frame_index_from_filename("frame_000042.png") == 42
    assert frame_index_from_filename("nodigits.jpg") == 0


def test_bbox_iou_identical_and_disjoint() -> None:
    assert bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert bbox_iou((0, 0, 10, 10), (50, 50, 10, 10)) == 0.0
    # Half-overlapping boxes: intersection 50, union 150.
    assert bbox_iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50 / 150)


def test_bbox_iou_degenerate_box_is_zero() -> None:
    assert bbox_iou((0, 0, 0, 10), (0, 0, 10, 10)) == 0.0


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
def test_dedup_merges_same_crack_across_frames() -> None:
    frames = [frame(i, observation(i, x=10 * i)) for i in range(5)]

    entities = deduplicate_observations(frames, iou_threshold=0.3, max_frame_gap=10)

    assert len(entities) == 1
    assert len(entities[0].observations) == 5
    # Union box spans the first and last observation.
    assert entities[0].box == (0, 0, 10 * 4 + 20, 20)


def test_dedup_keeps_distinct_cracks_separate() -> None:
    frames = [frame(i, observation(i, x=0), observation(i, x=60)) for i in range(3)]

    entities = deduplicate_observations(frames, iou_threshold=0.3, max_frame_gap=10)

    assert len(entities) == 2
    assert all(len(entity.observations) == 3 for entity in entities)


def test_dedup_never_merges_two_candidates_in_one_frame() -> None:
    # Two heavily overlapping observations in one frame, both matching the same
    # existing entity: only one may be absorbed, the other opens a new entity.
    frames = [frame(0, observation(0)), frame(1, observation(1, x=0), observation(1, x=2))]

    entities = deduplicate_observations(frames, iou_threshold=0.3, max_frame_gap=10)

    assert len(entities) == 2
    assert sorted(len(entity.observations) for entity in entities) == [1, 2]


def test_dedup_respects_max_frame_gap() -> None:
    frames = [frame(0, observation(0)), frame(50, observation(50))]

    assert len(deduplicate_observations(frames, iou_threshold=0.3, max_frame_gap=10)) == 2
    assert len(deduplicate_observations(frames, iou_threshold=0.3, max_frame_gap=100)) == 1


def test_dedup_is_deterministic_regardless_of_frame_order() -> None:
    frames = [frame(i, observation(i, x=3 * i)) for i in range(6)]
    shuffled = [frames[3], frames[0], frames[5], frames[2], frames[1], frames[4]]

    forward = [entity.box for entity in deduplicate_observations(frames)]
    backward = [entity.box for entity in deduplicate_observations(shuffled)]

    assert forward == backward


def test_dedup_separates_element_labels() -> None:
    frames = [
        frame(0, observation(0, label="column")),
        frame(1, observation(1, x=0, label="beam")),
    ]

    entities = deduplicate_observations(frames, iou_threshold=0.3, max_frame_gap=10)

    assert {entity.label for entity in entities} == {"column", "beam"}


# ---------------------------------------------------------------------------
# Observation extraction
# ---------------------------------------------------------------------------
def test_extract_crack_frames_reads_masks_and_metrics_sidecar(db_path: Path) -> None:
    out_dir = write_crack_job(
        db_path, "a" * 32, [crack_bar(10, 5), np.zeros(MASK_SHAPE, dtype=bool)]
    )

    frames = extract_crack_frames(out_dir)

    assert [item.frame_filename for item in frames] == ["frame_000000.jpg", "frame_000001.jpg"]
    assert len(frames[0].observations) == 1
    assert frames[0].observations[0].pixel_count == 36  # 12 x 3
    assert frames[1].observations == ()


def test_extract_crack_frames_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        extract_crack_frames(tmp_path / "nope")


def test_extract_element_frames_from_sidecar_json(tmp_path: Path) -> None:
    out_dir = tmp_path / "elements"
    out_dir.mkdir()
    (out_dir / "elements.json").write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "frame_filename": "frame_000000.jpg",
                        "elements": [
                            {
                                "label": "column",
                                "x": 4,
                                "y": 6,
                                "width": 10,
                                "height": 20,
                                "confidence": 0.9,
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    frames = extract_element_frames(out_dir)

    assert len(frames) == 1
    found = frames[0].observations[0]
    assert found.label == "column"
    assert found.confidence == pytest.approx(0.9)
    assert (found.x, found.y, found.width, found.height) == (4, 6, 10, 20)


def test_extract_element_frames_falls_back_to_class_masks(tmp_path: Path) -> None:
    out_dir = tmp_path / "elements"
    out_dir.mkdir()
    class_mask = np.zeros(MASK_SHAPE, dtype=np.uint8)
    class_mask[5:25, 5:25] = 1  # a "column" component
    assert cv2.imwrite(str(out_dir / "mask_000000.png"), class_mask)

    frames = extract_element_frames(out_dir)

    assert len(frames) == 1
    found = frames[0].observations[0]
    assert found.label == "column"
    assert found.pixel_count == 400


def test_extract_element_frames_ignores_unreadable_sidecar(tmp_path: Path) -> None:
    out_dir = tmp_path / "elements"
    out_dir.mkdir()
    (out_dir / "elements.json").write_text("{not json", encoding="utf-8")

    assert extract_element_frames(out_dir) == []


def test_extract_element_frames_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        extract_element_frames(tmp_path / "nope")


# ---------------------------------------------------------------------------
# Entity building
# ---------------------------------------------------------------------------
def test_build_crack_entities_aggregates_extrema_and_provenance() -> None:
    frames = [
        frame(0, observation(0, length_px=40.0, width_px=3.0, pixel_count=120)),
        frame(1, observation(1, x=2, length_px=60.0, width_px=5.0, pixel_count=300)),
    ]
    entities = deduplicate_observations(frames, iou_threshold=0.3, max_frame_gap=10)

    built = build_crack_entities(entities, include_observations=True)

    assert len(built) == 1
    crack = built[0]
    assert crack.defect_id == "CRK-0001"
    assert crack.observation_count == 2
    assert crack.length_px_max == pytest.approx(60.0)
    assert crack.length_px_mean == pytest.approx(50.0)
    assert crack.width_px_max == pytest.approx(5.0)
    assert crack.pixel_count_max == 300
    # The first observation opens the entity (no IoU); the second records its match.
    assert crack.observations[0].match_iou is None
    assert crack.observations[1].match_iou is not None
    # The representative is the strongest (longest) observation.
    assert crack.representative.length_px == pytest.approx(60.0)


def test_build_crack_entities_without_observations_keeps_representative() -> None:
    frames = [frame(0, observation(0, length_px=10.0)), frame(1, observation(1, length_px=90.0))]
    entities = deduplicate_observations(frames)

    built = build_crack_entities(entities, include_observations=False)

    assert len(built[0].observations) == 1
    assert built[0].observations[0].length_px == pytest.approx(90.0)


def test_build_element_entities_uses_element_id_field() -> None:
    # Regression: element entities must populate ``element_id`` (not the
    # crack-only ``defect_id``); found by the Phase 7 system evaluation once
    # fixtures finally produced non-empty element detections.
    frames = [
        frame(0, observation(0, label="column", confidence=0.80)),
        frame(1, observation(1, x=2, label="column", confidence=0.90)),
    ]
    entities = deduplicate_observations(frames, iou_threshold=0.3, max_frame_gap=10)

    built = build_element_entities(entities, include_observations=True)

    assert len(built) == 1
    element = built[0]
    assert element.element_id == "ELM-COLUMN-0001"
    assert element.kind == "element"
    assert element.observation_count == 2
    assert element.label == "column"
    assert element.confidence_max == pytest.approx(0.90)
    assert element.confidence_mean == pytest.approx(0.85)
    assert len(element.observations) == 2


def test_entities_are_ordered_by_first_seen_frame() -> None:
    # Three spatially separate cracks given out of chronological order.
    frames = [
        frame(2, observation(2, x=60)),
        frame(0, observation(0, x=0)),
        frame(1, observation(1, x=30)),
    ]
    entities = deduplicate_observations(frames, iou_threshold=0.3, max_frame_gap=10)

    built = build_crack_entities(entities)

    assert [crack.first_frame_index for crack in built] == [0, 1, 2]
    assert [crack.defect_id for crack in built] == ["CRK-0001", "CRK-0002", "CRK-0003"]


def test_mean_match_iou_is_none_for_single_observation_entity() -> None:
    accumulator = EntityAccumulator(label="crack")
    accumulator.add(observation(0), None)

    assert accumulator.mean_match_iou is None


# ---------------------------------------------------------------------------
# Job, persistence, and API
# ---------------------------------------------------------------------------
def test_run_aggregation_job_end_to_end(tmp_path: Path, db_path: Path) -> None:
    crack_job_id = "b" * 32
    # One crack drifting a few pixels per frame → six observations, one defect.
    write_crack_job(db_path, crack_job_id, [crack_bar(10 + i, 5) for i in range(6)])
    params = AggregationRequest(crack_job_id=crack_job_id)
    job_id = "c" * 32
    register_aggregation_job(db_path, job_id, params)

    run_aggregation_job(
        job_id, crack_job_id, params, output_root=tmp_path / "agg", db_path=db_path
    )

    with get_connection(db_path) as connection:
        record = connection.execute(
            "SELECT status, output_dir FROM processing_jobs WHERE id = ?", (job_id,)
        ).fetchone()
    assert record["status"] == JobStatus.COMPLETED.value
    out_dir = Path(record["output_dir"])
    registry_path = out_dir / "aggregated_defects.json"
    assert registry_path.is_file() and (out_dir / "manifest.json").is_file()

    registry = DefectRegistry.model_validate_json(registry_path.read_text(encoding="utf-8"))
    assert registry.summary.raw_crack_observations == 6
    assert registry.summary.unique_crack_defects == 1
    assert registry.summary.crack_dedup_ratio == pytest.approx(1 - 1 / 6)
    assert registry.cracks[0].observation_count == 6

    with get_connection(db_path) as connection:
        stored = connection.execute(
            "SELECT observation_count, provenance_json FROM global_crack_defects "
            "WHERE job_id = ?",
            (job_id,),
        ).fetchall()
    assert len(stored) == 1
    assert stored[0]["observation_count"] == 6
    assert len(json.loads(stored[0]["provenance_json"])) == 6


def test_run_aggregation_job_rejects_incomplete_source(db_path: Path, tmp_path: Path) -> None:
    params = AggregationRequest(crack_job_id="d" * 32)
    job_id = "e" * 32
    register_aggregation_job(db_path, job_id, params)

    run_aggregation_job(
        job_id, params.crack_job_id, params, output_root=tmp_path / "agg", db_path=db_path
    )

    with get_connection(db_path) as connection:
        record = connection.execute(
            "SELECT status, error FROM processing_jobs WHERE id = ?", (job_id,)
        ).fetchone()
    assert record["status"] == JobStatus.FAILED.value
    assert "Unknown crack_mapping job" in record["error"]


def test_load_aggregation_run_returns_none_for_unknown_job(db_path: Path) -> None:
    assert load_aggregation_run("f" * 32, db_path=db_path) is None


def test_aggregate_endpoint_returns_404_for_unknown_crack_job(client: TestClient) -> None:
    response = client.post("/api/processing/aggregate-results", json={"crack_job_id": "0" * 32})

    assert response.status_code == 404


def test_aggregate_endpoint_runs_and_exposes_registry(client: TestClient) -> None:
    # The client talks to the default database, so the source job is seeded there.
    crack_job_id = "1" * 32
    write_crack_job(None, crack_job_id, [crack_bar(10, 5), crack_bar(10, 5)])

    response = client.post(
        "/api/processing/aggregate-results", json={"crack_job_id": crack_job_id}
    )

    assert response.status_code == 202
    body = response.json()
    job_id = body["job_id"]
    assert body["status"] == JobStatus.PENDING.value
    assert body["summary"] is None

    fetched = client.get(f"/api/processing/aggregations/{job_id}")
    assert fetched.status_code == 200
    result = fetched.json()
    assert result["status"] == JobStatus.COMPLETED.value
    assert result["summary"]["raw_crack_observations"] == 2
    assert result["summary"]["unique_crack_defects"] == 1
    assert result["cracks"][0]["defect_id"] == "CRK-0001"
    assert result["cracks"][0]["observation_count"] == 2
    # Provenance is included by default, so the UI can link a defect to its frames.
    assert len(result["cracks"][0]["observations"]) == 2


def test_aggregate_endpoint_returns_409_for_pending_source(client: TestClient) -> None:
    pending_id = create_job_record(
        job_type="crack_mapping", video_filename="phase6_video", params_json="{}"
    )

    response = client.post("/api/processing/aggregate-results", json={"crack_job_id": pending_id})

    assert response.status_code == 409


def test_get_aggregation_returns_404_for_unknown_id(client: TestClient) -> None:
    assert client.get("/api/processing/aggregations/does-not-exist").status_code == 404
