"""AeroTwin AI — automated end-to-end system evaluation (Phase 7).

Runs synthetic UAV data through the complete production pipeline and writes
performance telemetry to ``docs/sample_outputs/`` for the thesis results
chapter:

    video import → frame extraction → AI analysis (material segmentation,
    element detection, crack mapping) → cross-frame aggregation
    → seismic assessment → report generation

Telemetry captured per stage: wall time, backend ``processing_time_seconds``,
inference runtimes (``mean_frame_inference_seconds``), dedup ratios
(``crack_dedup_ratio``, raw vs. unique observations), seismic score/factors,
and report file sizes.

Run from the ``backend/`` directory:

    python tests/evaluate_system.py [--seed 42] [--keep] [--data-dir PATH]

Isolation & determinism (domain rule 3): all RNG seeds are fixed; the pipeline
runs inside a temporary ``AEROTWIN_DATA_DIR`` so the repository's real
``data/`` is never touched. Model weights: real ``.pt`` files from
``backend/app/models/weights/`` are used when present; otherwise deterministic
random-init fixtures are generated and flagged as
``weights.source == "synthetic_random_init"`` — timings, dedup mechanics, and
scoring remain valid, but task-level accuracy figures would not be.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import shutil
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap — runs BEFORE any ``app.*`` import (Settings are read at import).
# ---------------------------------------------------------------------------
BACKEND_DIR: Path = Path(__file__).resolve().parents[1]
REPO_ROOT: Path = BACKEND_DIR.parent
DOCS_SAMPLES_DIR: Path = REPO_ROOT / "docs" / "sample_outputs"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Console output must survive non-UTF-8 Windows consoles/redirects (cp1252
# cannot encode "-> / em-dashes" in log lines when stdout is piped).
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - stream may be closed/unsupported
            pass


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AeroTwin AI end-to-end system evaluation (Phase 7)."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Pipeline data dir (default: fresh temp dir; the repo data/ is never touched).",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep the temporary data/weights directories after the run.",
    )
    return parser.parse_args(argv)


ARGS = _parse_args()
SEED: int = ARGS.seed
STARTED_AT = datetime.now(timezone.utc)
RUN_STAMP = STARTED_AT.strftime("%Y%m%dT%H%M%SZ")

#: True when we own the temp data dir (safe to delete after the run).
_PREEXISTING_DATA_DIR = "AEROTWIN_DATA_DIR" in os.environ
if ARGS.data_dir is not None:
    os.environ["AEROTWIN_DATA_DIR"] = str(ARGS.data_dir.resolve())
elif not _PREEXISTING_DATA_DIR:
    os.environ["AEROTWIN_DATA_DIR"] = str(
        Path(tempfile.gettempdir()) / f"aerotwin_eval_{RUN_STAMP}"
    )
_OWN_TEMP_DATA_DIR = ARGS.data_dir is None and not _PREEXISTING_DATA_DIR

REAL_WEIGHTS_DIR: Path = BACKEND_DIR / "app" / "models" / "weights"
WEIGHTS_FILENAMES: tuple[str, ...] = (
    "material_model.pt",
    "element_model.pt",
    "crack_model.pt",
)


def _real_weights_available() -> bool:
    return all((REAL_WEIGHTS_DIR / name).is_file() for name in WEIGHTS_FILENAMES)


#: When real weights are missing, generate deterministic fixtures in a temp
#: dir and point the app at it via ``AEROTWIN_WEIGHTS_DIR`` (read at import).
SYNTHETIC_WEIGHTS_DIR: Path | None = None
if "AEROTWIN_WEIGHTS_DIR" not in os.environ and not _real_weights_available():
    SYNTHETIC_WEIGHTS_DIR = Path(tempfile.mkdtemp(prefix="aerotwin_eval_weights_"))
    os.environ["AEROTWIN_WEIGHTS_DIR"] = str(SYNTHETIC_WEIGHTS_DIR)

if SYNTHETIC_WEIGHTS_DIR is not None:
    WEIGHTS_SOURCE = "synthetic_random_init"
elif _real_weights_available():
    WEIGHTS_SOURCE = "real_repository_weights"
else:
    WEIGHTS_SOURCE = "external_env_weights"

# ---------------------------------------------------------------------------
# Third-party + application imports (after env bootstrap)
# ---------------------------------------------------------------------------
try:
    import cv2
    import numpy as np
    import torch
except ImportError as exc:  # pragma: no cover - environment problem, not logic
    raise SystemExit(
        f"[eval] missing evaluation dependency: {exc}. "
        "Install with: pip install -r backend/requirements.txt"
    ) from exc

from app.core.config import settings  # noqa: E402
from app.core.database import (  # noqa: E402
    create_job_record,
    get_connection,
    get_job_record,
    init_database,
)
from app.models.base import MiniUNet  # noqa: E402
from app.models.crack_detector import (  # noqa: E402
    CRACK_CLASSES,
    CRACK_WEIGHTS_FILENAME,
    CrackDetector,
)
from app.models.material_segmenter import (  # noqa: E402
    MATERIAL_CLASSES,
    MATERIAL_WEIGHTS_FILENAME,
    run_material_segmentation_job,
)
from app.models.structural_element_detector import (  # noqa: E402
    ELEMENT_CLASSES,
    ELEMENT_WEIGHTS_FILENAME,
    StructuralElementDetector,
    run_element_detection_job,
)
from app.schemas.assessment import (  # noqa: E402
    BuildingProfileCreate,
    CodeCompliance,
    ConstructionType,
    SeismicCalculationRequest,
)
from app.schemas.defect_registry import AggregationRequest  # noqa: E402
from app.schemas.processing import (  # noqa: E402
    CrackMappingRequest,
    ElementDetectionRequest,
    FrameExtractionRequest,
    JobStatus,
    MaterialSegmentationRequest,
)
from app.schemas.reports import ReportFormat, ReportGenerateRequest  # noqa: E402
from app.services.crack_mapper import run_crack_mapping_job  # noqa: E402
from app.services.cross_frame_aggregator import (  # noqa: E402
    JOB_TYPE_AGGREGATION,
    run_aggregation_job,
)
from app.services.report_generator import (  # noqa: E402
    create_report,
    get_report,
    run_report_generation,
)
from app.services.seismic_calculator import (  # noqa: E402
    create_assessment,
    create_building_profile,
    get_assessment,
    run_seismic_assessment_job,
)
from app.services.uav_preprocessor import (  # noqa: E402
    JOB_TYPE_FRAME_EXTRACTION,
    ingest_video,
    run_frame_extraction_job,
)

# ---------------------------------------------------------------------------
# Run state
# ---------------------------------------------------------------------------
STAGES: list[dict] = []
OUTPUTS: dict = {}


@contextmanager
def stage(stage_id: str, description: str):
    """Time a pipeline stage and always record an entry (failed on exception)."""
    print(f"[eval] {stage_id}: {description}", flush=True)
    entry: dict = {
        "stage": stage_id,
        "description": description,
        "status": "running",
        "wall_time_seconds": 0.0,
    }
    STAGES.append(entry)
    started = time.perf_counter()
    try:
        yield entry
    except BaseException:
        entry["status"] = "failed"
        raise
    finally:
        entry["wall_time_seconds"] = round(time.perf_counter() - started, 4)


def _finalize_job_stage(entry: dict, job_id: str) -> dict:
    """Copy job telemetry into the stage entry; fail loudly if the job failed.

    Job runners never raise — errors land on the DB row — so the evaluator
    translates a failed row into a script failure (thesis runs are atomic).
    """
    row = get_job_record(job_id)
    if row is None:
        raise SystemExit(f"[eval] job {job_id} vanished from processing_jobs")
    metrics = json.loads(row["metrics_json"]) if row["metrics_json"] else {}
    entry.update(
        {
            "job_id": job_id,
            "job_status": row["status"],
            "processing_time_seconds": row["processing_time_seconds"],
            "frames_written": row["frames_written"],
            "output_dir": row["output_dir"],
            "metrics": metrics,
        }
    )
    if row["status"] != JobStatus.COMPLETED.value:
        entry["error"] = row["error"]
        raise SystemExit(
            f"[eval] stage '{entry['stage']}' failed: {row['error'] or 'unknown error'}"
        )
    entry["status"] = "completed"
    return {"job_id": job_id, "metrics": metrics, "output_dir": row["output_dir"]}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


#: Head scales used by the synthetic fixtures (fixed; biases are calibrated).
CRACK_HEAD_SCALE: float = 4.0
ELEMENT_HEAD_SCALE: float = 4.0
MATERIAL_HEAD_SCALE: float = 4.0


def _model_state(
    num_classes: int, seed: int, head_scale: float, background_bias: float
) -> dict:
    """MiniUNet state_dict with a scaled 1x1 head and tuned background bias."""
    torch.manual_seed(seed)
    model = MiniUNet(in_channels=3, num_classes=num_classes, base_channels=16)
    state = model.state_dict()
    if "head.weight" in state:
        state["head.weight"] = state["head.weight"] * head_scale
    if "head.bias" in state:
        bias = state["head.bias"].clone()
        bias[0] = background_bias
        state["head.bias"] = bias
    return state


def _write_synthetic_weights(directory: Path, seed: int) -> dict:
    """Deterministic fixtures for all three models with calibrated heads.

    A plain random init produces *binary* masks (0% or ~100% detections: the
    random head margins are nearly constant across pixels), so the 1x1 head is
    calibrated instead: ``head.weight`` is scaled up and ``head.bias[0]``
    (class 0 = background) tuned against the real detector paths to land in a
    sparse operating band (~2% non-background pixels, a couple of components
    per frame). That exercises tiled inference, component filtering, Zhang-Suen
    thinning, and cross-frame dedup realistically.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    directory.mkdir(parents=True, exist_ok=True)
    # (filename, num_classes, head_weight_scale, background_logit_bias)
    # Initial values; crack/element are re-calibrated on the *extracted* frames
    # once stage 2 completes (mp4/JPEG round-trips shift the logit margins).
    specs = (
        (MATERIAL_WEIGHTS_FILENAME, len(MATERIAL_CLASSES), MATERIAL_HEAD_SCALE, 0.10),
        (ELEMENT_WEIGHTS_FILENAME, len(ELEMENT_CLASSES), ELEMENT_HEAD_SCALE, 0.15),
        (CRACK_WEIGHTS_FILENAME, len(CRACK_CLASSES), CRACK_HEAD_SCALE, -0.10),
    )
    written: dict[str, dict] = {}
    for filename, num_classes, head_scale, background_bias in specs:
        path = directory / filename
        torch.save(
            _model_state(num_classes, seed, head_scale, background_bias), path
        )
        written[filename] = {"size_bytes": path.stat().st_size, "sha256": _sha256(path)}
    return written


def _synthesize_video(
    target: Path,
    *,
    seed: int,
    frames: int = 24,
    fps: int = 10,
    size: tuple[int, int] = (64, 48),
) -> dict:
    """Paint a small facade-like clip: window grid, texture, drifting crack line.

    The polyline jitters per frame so cross-frame aggregation sees genuinely
    overlapping observations (like a real low-altitude UAV video).
    """
    width, height = size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(target), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise SystemExit(
            "[eval] cv2.VideoWriter could not open an mp4v stream (codec unavailable)."
        )
    rng = np.random.default_rng(seed)
    gradient = np.linspace(0, 70, width, dtype=np.int16)[None, :, None]
    base = np.clip(
        np.full((height, width, 3), 55, dtype=np.int16) + gradient, 0, 255
    ).astype(np.uint8)
    try:
        for index in range(frames):
            frame = base.copy()
            # Window grid (structural texture).
            for gx in range(4, width - 8, 14):
                for gy in range(4, height // 2, 12):
                    cv2.rectangle(frame, (gx, gy), (gx + 7, gy + 6), (35, 40, 48), -1)
            # Drifting crack-like polyline with per-frame jitter.
            jitter = rng.integers(-1, 2, size=2)
            points = np.array(
                [
                    [6 + int(jitter[0]), 5 + int(jitter[1])],
                    [width // 2 + int(jitter[0]), height // 3 + int(jitter[1])],
                    [width - 7 + int(jitter[0]), height - 6 + int(jitter[1])],
                ],
                dtype=np.int32,
            )
            cv2.polylines(frame, [points], isClosed=False, color=(25, 25, 30), thickness=1)
            # Slow pan + sensor noise (motion between frames).
            frame = np.roll(frame, index % 3, axis=1)
            noise = rng.integers(0, 14, size=frame.shape, dtype=np.uint8)
            frame = np.clip(
                frame.astype(np.int16) + noise.astype(np.int16), 0, 255
            ).astype(np.uint8)
            writer.write(frame)
    finally:
        writer.release()
    return {
        "path": str(target),
        "size_bytes": target.stat().st_size,
        "frames": frames,
        "fps": fps,
        "width": width,
        "height": height,
    }


def _calibrate_synthetic_weights(frames_dir: Path) -> dict:
    """Re-fit crack/element heads on the *actual* extracted frames.

    The mp4 encode + decode + JPEG write shifts pixel statistics relative to
    the in-memory painting, and the synthetic-head decision margins are sharp,
    so masks silently go to 0% or ~100%. A deterministic bias grid is therefore
    scored with the real detectors until detections are sparse but non-zero,
    guaranteeing meaningful dedup/element telemetry. The chosen operating point
    is recorded in the evaluation output.
    """
    paths = sorted(
        [path for ext in ("*.jpg", "*.jpeg", "*.png") for path in frames_dir.glob(ext)]
    )[:3]
    frames = [image for path in paths if (image := cv2.imread(str(path))) is not None]
    if not frames:
        return {"status": "skipped", "reason": "no extracted frames found"}

    weights_dir = settings.weights_dir
    total_pixels = sum(int(frame.shape[0] * frame.shape[1]) for frame in frames)
    result: dict = {"status": "calibrated", "frames_used": len(frames)}

    # --- crack head: target 0.3%-10% crack pixels with >=1 component --------
    crack_path = weights_dir / CRACK_WEIGHTS_FILENAME
    evaluations: list[dict] = []
    for index in range(1, 16):  # bias -0.025 .. -0.375
        bias = round(-0.025 * index, 4)
        torch.save(
            _model_state(len(CRACK_CLASSES), SEED, CRACK_HEAD_SCALE, bias), crack_path
        )
        detector = CrackDetector(weights_path=crack_path)
        pixels = components = 0
        for frame in frames:
            mask = detector.predict(frame)
            pixels += int(mask.sum())
            count, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
            components += count - 1
        evaluations.append(
            {
                "head_scale": CRACK_HEAD_SCALE,
                "background_bias": bias,
                "crack_pixel_fraction": round(pixels / max(total_pixels, 1), 5),
                "components": components,
            }
        )
    eligible = [entry for entry in evaluations if entry["components"] > 0]
    chosen = next(
        (entry for entry in eligible if entry["crack_pixel_fraction"] >= 0.003),
        eligible[0] if eligible else evaluations[0],
    )
    torch.save(
        _model_state(len(CRACK_CLASSES), SEED, CRACK_HEAD_SCALE, chosen["background_bias"]),
        crack_path,
    )
    result["crack"] = chosen

    # --- element head: target >=1 detected element per sampled frame --------
    element_path = weights_dir / ELEMENT_WEIGHTS_FILENAME
    evaluations = []
    for index in range(1, 18):  # bias +0.325 .. -0.10
        bias = round(0.35 - 0.025 * index, 4)
        torch.save(
            _model_state(len(ELEMENT_CLASSES), SEED, ELEMENT_HEAD_SCALE, bias), element_path
        )
        detector = StructuralElementDetector(weights_path=element_path)
        elements = 0
        for frame in frames:
            _, detected = detector.detect(frame)
            elements += len(detected)
        evaluations.append(
            {
                "head_scale": ELEMENT_HEAD_SCALE,
                "background_bias": bias,
                "elements": elements,
                "elements_per_frame": round(elements / len(frames), 2),
            }
        )
    eligible = [
        entry
        for entry in evaluations
        if len(frames) <= entry["elements"] <= 12 * len(frames)
    ]
    chosen = next(
        (entry for entry in eligible),
        next((entry for entry in evaluations if entry["elements"] > 0), evaluations[0]),
    )
    torch.save(
        _model_state(
            len(ELEMENT_CLASSES), SEED, ELEMENT_HEAD_SCALE, chosen["background_bias"]
        ),
        element_path,
    )
    result["element"] = chosen

    # Refresh fixture hashes (files were rewritten during calibration).
    if SYNTHETIC_WEIGHTS_DIR is not None:
        OUTPUTS["synthetic_weights"] = {
            name: {
                "size_bytes": (weights_dir / name).stat().st_size,
                "sha256": _sha256(weights_dir / name),
            }
            for name in WEIGHTS_FILENAMES
            if (weights_dir / name).is_file()
        }
    return result


def _run_pipeline() -> None:
    """Drive every pipeline stage synchronously and collect telemetry."""
    # -- Stage 1: video import ------------------------------------------------
    source_dir = Path(tempfile.gettempdir()) / f"aerotwin_eval_src_{RUN_STAMP}"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_path = source_dir / "synthetic_facade.mp4"
    with stage(
        "video_import", "Synthesize UAV clip and ingest into data/inputs/raw_videos/"
    ) as entry:
        video_info = _synthesize_video(source_path, seed=SEED)
        record = ingest_video(source_path)
        entry["status"] = "completed"
        entry["metrics"] = {
            "source_clip": video_info,
            "ingested_video": record.model_dump(mode="json"),
        }
        OUTPUTS["video"] = record.model_dump(mode="json")
    shutil.rmtree(source_dir, ignore_errors=True)

    # -- Stage 2: frame extraction -------------------------------------------
    extraction_params = FrameExtractionRequest(filename=record.filename, frame_step=3)
    extraction_job_id = create_job_record(
        job_type=JOB_TYPE_FRAME_EXTRACTION,
        video_filename=record.filename,
        video_id=record.video_id,
        params_json=extraction_params.model_dump_json(),
    )
    with stage(
        "frame_extraction", "Extract frames from the ingested UAV video"
    ) as entry:
        run_frame_extraction_job(
            extraction_job_id, Path(record.stored_path), extraction_params
        )
        job = _finalize_job_stage(entry, extraction_job_id)
    frames_run_dir = Path(job["output_dir"])
    OUTPUTS["frames_run_dir"] = str(frames_run_dir)

    #: Relative path (to data/processed/frames/) the analysis jobs consume.
    frames_rel = frames_run_dir.relative_to(settings.frames_dir).as_posix()

    # -- Stage 2b: calibrate synthetic weights on the real decoded frames ----
    if SYNTHETIC_WEIGHTS_DIR is not None:
        with stage(
            "weights_calibration",
            "Score a deterministic head-bias grid on decoded frames (real detectors)",
        ) as entry:
            OUTPUTS["weight_calibration"] = _calibrate_synthetic_weights(frames_run_dir)
            entry["status"] = "completed"
            entry["metrics"] = OUTPUTS["weight_calibration"]

    # -- Stage 3: AI analysis (material → elements → cracks) -----------------
    material_params = MaterialSegmentationRequest(frames_path=frames_rel)
    material_job_id = create_job_record(
        job_type="material_segmentation",
        video_filename=record.filename,
        video_id=record.video_id,
        params_json=material_params.model_dump_json(),
    )
    with stage(
        "material_segmentation", "Tiled MiniUNet inference: masks per frame"
    ) as entry:
        run_material_segmentation_job(
            material_job_id, frames_run_dir, material_params
        )
        OUTPUTS["material_job"] = _finalize_job_stage(entry, material_job_id)

    element_params = ElementDetectionRequest(frames_path=frames_rel)
    element_job_id = create_job_record(
        job_type="element_detection",
        video_filename=record.filename,
        video_id=record.video_id,
        params_json=element_params.model_dump_json(),
    )
    with stage(
        "element_detection", "Detect structural elements (column/beam/wall)"
    ) as entry:
        run_element_detection_job(element_job_id, frames_run_dir, element_params)
        OUTPUTS["element_job"] = _finalize_job_stage(entry, element_job_id)

    crack_params = CrackMappingRequest(frames_path=frames_rel)
    crack_job_id = create_job_record(
        job_type="crack_mapping",
        video_filename=record.filename,
        video_id=record.video_id,
        params_json=crack_params.model_dump_json(),
    )
    with stage(
        "crack_mapping", "Detect + skeletonize cracks; log frame defect metrics"
    ) as entry:
        run_crack_mapping_job(crack_job_id, frames_run_dir, crack_params)
        OUTPUTS["crack_job"] = _finalize_job_stage(entry, crack_job_id)

    # -- Stage 4: cross-frame aggregation ------------------------------------
    aggregation_params = AggregationRequest(
        crack_job_id=crack_job_id, element_job_id=element_job_id
    )
    aggregation_job_id = create_job_record(
        job_type=JOB_TYPE_AGGREGATION,
        video_filename=record.filename,
        video_id=record.video_id,
        params_json=aggregation_params.model_dump_json(),
    )
    with stage(
        "cross_frame_aggregation",
        "Deduplicate frame-local detections into the global defect registry",
    ) as entry:
        run_aggregation_job(
            aggregation_job_id, crack_job_id, aggregation_params
        )
        OUTPUTS["aggregation_job"] = _finalize_job_stage(entry, aggregation_job_id)

    OUTPUTS["crack_job_id"] = crack_job_id
    OUTPUTS["element_job_id"] = element_job_id


def _reportlab_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("reportlab") is not None


def _run_assessment_and_report() -> None:
    """Stages 5-6: seismic scoring on registry defects, then report files."""
    crack_job_id = OUTPUTS["crack_job_id"]
    element_job_id = OUTPUTS["element_job_id"]

    # -- Stage 5: seismic assessment -----------------------------------------
    profile = create_building_profile(
        BuildingProfileCreate(
            name=f"Synthetic Evaluation Tower {RUN_STAMP}",
            structure_age_years=25.0,
            construction_type=ConstructionType.RC_FRAME,
            num_stories=6,
            code_compliance=CodeCompliance.PARTIALLY_COMPLIANT,
            notes="Phase 7 automated end-to-end evaluation (synthetic data).",
        )
    )
    assessment_params = SeismicCalculationRequest(
        profile_id=profile.id,
        crack_job_id=crack_job_id,
        element_job_id=element_job_id,
    )
    assessment_id = create_assessment(profile, assessment_params)
    with stage(
        "seismic_assessment", "Deterministic weighted seismic vulnerability score"
    ) as entry:
        run_seismic_assessment_job(assessment_id, assessment_params)
        assessment = get_assessment(assessment_id)
        if assessment is None:
            raise SystemExit(f"[eval] assessment {assessment_id} vanished from DB")
        if assessment.status != JobStatus.COMPLETED.value:
            entry["status"] = "failed"
            entry["error"] = assessment.error
            raise SystemExit(
                "[eval] stage 'seismic_assessment' failed: "
                f"{assessment.error or 'unknown error'}"
            )
        entry["status"] = "completed"
        entry["metrics"] = {
            "vulnerability_score": assessment.vulnerability_score,
            "classification": assessment.classification,
        }
        OUTPUTS["assessment"] = assessment.model_dump(mode="json")
    OUTPUTS["assessment_id"] = assessment_id

    # -- Stage 6: report generation ------------------------------------------
    report_format = ReportFormat.BOTH if _reportlab_available() else ReportFormat.JSON
    report_id = create_report(assessment_id, report_format)
    report_request = ReportGenerateRequest(
        assessment_id=assessment_id, report_format=report_format
    )
    with stage("report_generation", "Compile JSON + PDF inspection report") as entry:
        run_report_generation(report_id, report_request)
        report = get_report(report_id)
        if report is None:
            raise SystemExit(f"[eval] report {report_id} vanished from DB")
        if report.status != JobStatus.COMPLETED.value:
            entry["status"] = "failed"
            entry["error"] = report.error
            raise SystemExit(
                f"[eval] stage 'report_generation' failed: "
                f"{report.error or 'unknown error'}"
            )
        entry["status"] = "completed"
        entry["metrics"] = {
            "report_format": report.report_format.value
            if hasattr(report.report_format, "value")
            else str(report.report_format),
            "json_path": report.json_path,
            "pdf_path": report.pdf_path,
            "size_bytes": report.size_bytes,
        }
        OUTPUTS["report"] = report.model_dump(mode="json")
    OUTPUTS["report_id"] = report_id


def _package_versions() -> dict[str, str | None]:
    from importlib import metadata

    versions: dict[str, str | None] = {}
    for distribution in (
        "torch",
        "opencv-python",
        "opencv-python-headless",
        "numpy",
        "fastapi",
        "uvicorn",
        "pydantic",
        "starlette",
        "reportlab",
        "pytest",
        "pyinstaller",
    ):
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _weights_info() -> dict:
    files: dict[str, dict | None] = {}
    for name in WEIGHTS_FILENAMES:
        path = settings.weights_dir / name
        files[name] = (
            {"size_bytes": path.stat().st_size, "sha256": _sha256(path)}
            if path.is_file()
            else None
        )
    return {"source": WEIGHTS_SOURCE, "directory": str(settings.weights_dir), "files": files}


def _db_counts() -> dict[str, int | None]:
    counts: dict[str, int | None] = {}
    tables = (
        "videos",
        "processing_jobs",
        "frame_defect_metrics",
        "building_profiles",
        "assessments",
        "reports",
        "global_crack_defects",
        "global_element_instances",
    )
    try:
        with get_connection(settings.database_path) as connection:
            for table in tables:
                try:
                    counts[table] = connection.execute(
                        f"SELECT COUNT(*) FROM {table}"
                    ).fetchone()[0]
                except Exception:  # noqa: BLE001 - table may not exist yet
                    counts[table] = None
    except Exception:  # noqa: BLE001 - DB may not exist on failed runs
        for table in tables:
            counts.setdefault(table, None)
    return counts


def _aggregation_summary() -> dict:
    metrics = OUTPUTS.get("aggregation_job", {}).get("metrics", {}) or {}
    keys = (
        "raw_crack_observations",
        "unique_crack_defects",
        "crack_dedup_ratio",
        "crack_dedup_reduction",
        "mean_observations_per_crack",
        "raw_element_observations",
        "unique_element_instances",
        "frames_analyzed",
        "iou_threshold",
        "max_frame_gap",
    )
    return {key: metrics.get(key) for key in keys}


def _key_metric(entry: dict) -> str:
    """Compact per-stage metric for the Markdown results table."""
    metrics = entry.get("metrics") or {}
    stage_id = entry["stage"]
    if stage_id == "video_import":
        clip = metrics.get("source_clip", {}) or {}
        return (
            f"{clip.get('frames')}f @ {clip.get('fps')}fps "
            f"{clip.get('width')}x{clip.get('height')}"
        )
    if stage_id in ("material_segmentation", "element_detection"):
        mean = metrics.get("mean_frame_inference_seconds")
        total = metrics.get("total_inference_seconds")
        if mean is not None:
            if total is not None:
                return f"mean {float(mean) * 1000:.1f} ms/frame (total {total:.3f} s)"
            return f"mean {float(mean) * 1000:.1f} ms/frame"
    if stage_id == "frame_extraction":
        return f"{entry.get('frames_written')} frames written"
    if stage_id == "element_detection":
        return f"{metrics.get('total_elements', '—')} elements"
    if stage_id == "crack_mapping":
        frames = metrics.get("frames_with_cracks", metrics.get("frames_processed"))
        if frames is not None:
            return f"{frames} frames w/ cracks"
    if stage_id == "cross_frame_aggregation":
        ratio = metrics.get("crack_dedup_ratio")
        if ratio is not None:
            return (
                f"dedup {float(ratio) * 100:.1f}% "
                f"({metrics.get('raw_crack_observations')}→"
                f"{metrics.get('unique_crack_defects')} crack obs)"
            )
    if stage_id == "seismic_assessment":
        score = metrics.get("vulnerability_score")
        if score is not None:
            return f"score {score}/100 — {metrics.get('classification')}"
    if stage_id == "report_generation" and metrics.get("size_bytes") is not None:
        return f"{metrics.get('size_bytes')} bytes written"
    compact = json.dumps(metrics, default=str)
    return compact if len(compact) <= 96 else compact[:93] + "..."


def _build_telemetry(outcome: str, error_message: str | None) -> dict:
    finished = datetime.now(timezone.utc)
    total_stage_seconds = round(
        sum(entry.get("wall_time_seconds", 0.0) for entry in STAGES), 4
    )
    notes = [
        "Seeded run: random/NumPy/PyTorch RNGs fixed (seed="
        f"{SEED}) for reproducibility (domain rule 3).",
        f"Pipeline ran in an isolated data dir: {settings.data_dir}.",
    ]
    if WEIGHTS_SOURCE == "synthetic_random_init":
        notes.append(
            "Model weights were NOT present in backend/app/models/weights/; "
            "deterministic random-init fixtures were used. Inference runtimes, "
            "dedup ratios, and scoring mechanics are valid, but task-level "
            "accuracy (IoU/F1/score calibration) would NOT be."
        )
    if not _reportlab_available():
        notes.append("reportlab missing: report fell back to JSON only.")
    return {
        "schema": "aerotwin.system_evaluation.v1",
        "outcome": outcome,
        "error": error_message,
        "generated_at": finished.isoformat(),
        "started_at": STARTED_AT.isoformat(),
        "run_stamp": RUN_STAMP,
        "seed": SEED,
        "environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "machine": platform.machine(),
            "device": "cpu",
            "packages": _package_versions(),
        },
        "weights": _weights_info(),
        "pipeline": {
            "stages": STAGES,
            "stage_wall_time_total_seconds": total_stage_seconds,
            "outcomes": {
                key: value
                for key, value in OUTPUTS.items()
                if key
                in ("video", "frames_run_dir", "assessment_id", "report_id", "report")
            },
        },
        "aggregation": _aggregation_summary(),
        "assessment": OUTPUTS.get("assessment") or {},
        "database": {"path": str(settings.database_path), "rows": _db_counts()},
        "notes": notes,
    }


def _write_telemetry(telemetry: dict) -> tuple[Path, Path]:
    """Persist JSON telemetry + a Markdown results table for the thesis."""
    DOCS_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    json_path = DOCS_SAMPLES_DIR / "system_evaluation.json"
    json_path.write_text(
        json.dumps(telemetry, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )

    lines: list[str] = []
    lines.append("# AeroTwin AI — System Evaluation (Phase 7)")
    lines.append("")
    lines.append(f"- **Generated:** {telemetry['generated_at']}")
    lines.append(f"- **Outcome:** `{telemetry['outcome']}`")
    if telemetry.get("error"):
        lines.append(f"- **Error:** `{telemetry['error']}`")
    lines.append(f"- **Seed:** {telemetry['seed']}")
    lines.append(
        f"- **Stage wall-time total:** "
        f"{telemetry['pipeline']['stage_wall_time_total_seconds']} s"
    )
    lines.append(f"- **Weights source:** `{telemetry['weights']['source']}`")
    lines.append(f"- **Data dir:** `{settings.data_dir}`")
    lines.append("")
    lines.append("## Pipeline stages")
    lines.append("")
    lines.append("| # | Stage | Status | Wall (s) | Backend (s) | Key metric |")
    lines.append("|---|-------|--------|---------:|------------:|------------|")
    for index, entry in enumerate(telemetry["pipeline"]["stages"], start=1):
        backend_time = entry.get("processing_time_seconds")
        backend_cell = (
            f"{backend_time:.3f}" if isinstance(backend_time, (int, float)) else "—"
        )
        lines.append(
            f"| {index} | `{entry['stage']}` | {entry['status']} "
            f"| {entry['wall_time_seconds']:.3f} | {backend_cell} "
            f"| {_key_metric(entry)} |"
        )
    lines.append("")
    lines.append("## Cross-frame dedup")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|------:|")
    for key, value in telemetry["aggregation"].items():
        lines.append(f"| {key} | {value} |")
    lines.append("")
    assessment = telemetry["assessment"]
    lines.append("## Seismic assessment")
    lines.append("")
    lines.append(
        f"- **Vulnerability score:** {assessment.get('vulnerability_score')}/100 "
        f"— **{assessment.get('classification')}**"
    )
    factors = (assessment.get("result") or {}).get("factors") or []
    if factors:
        lines.append("")
        lines.append("| Factor | Value | Weight | Contribution |")
        lines.append("|--------|------:|-------:|-------------:|")
        for factor in factors:
            lines.append(
                f"| {factor.get('name')} | {factor.get('value')} "
                f"| {factor.get('weight')} | {factor.get('contribution')} |"
            )
    lines.append("")
    report = telemetry["pipeline"]["outcomes"].get("report") or {}
    lines.append("## Report artifacts")
    lines.append("")
    lines.append(f"- JSON: `{report.get('json_path')}`")
    if report.get("pdf_path"):
        lines.append(f"- PDF: `{report.get('pdf_path')}`")
    lines.append(f"- Size: {report.get('size_bytes')} bytes")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    for note in telemetry["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    md_path = DOCS_SAMPLES_DIR / "system_evaluation.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def main() -> int:
    """Run the full evaluation, write telemetry, clean up, return an exit code."""
    print(
        f"[eval] AeroTwin system evaluation | seed={SEED} "
        f"| data={os.environ.get('AEROTWIN_DATA_DIR')}",
        flush=True,
    )
    overall_started = time.perf_counter()

    if SYNTHETIC_WEIGHTS_DIR is not None:
        OUTPUTS["synthetic_weights"] = _write_synthetic_weights(
            SYNTHETIC_WEIGHTS_DIR, SEED
        )
        print(
            f"[eval] real weights absent → synthetic fixtures in "
            f"{SYNTHETIC_WEIGHTS_DIR} (source={WEIGHTS_SOURCE})",
            flush=True,
        )
    # Seed app-level RNGs as well (services read settings.random_seed).
    random.seed(SEED)
    np.random.seed(SEED % (2**32))
    torch.manual_seed(SEED)

    # Mirror the FastAPI lifespan: create data dirs + WAL schema (idempotent).
    settings.ensure_directories()
    init_database()

    outcome, error_message = "completed", None
    try:
        _run_pipeline()
        _run_assessment_and_report()
    except SystemExit as exc:  # stage failure (job row error) — telemetry still written
        outcome, error_message = "failed", str(exc)
        print(str(exc), file=sys.stderr, flush=True)
    except Exception as exc:  # noqa: BLE001 - evaluation boundary: report everything
        outcome = "failed"
        error_message = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()

    telemetry = _build_telemetry(outcome, error_message)
    telemetry["overall_wall_time_seconds"] = round(time.perf_counter() - overall_started, 4)
    json_path, md_path = _write_telemetry(telemetry)

    print(f"[eval] telemetry: {json_path}", flush=True)
    print(f"[eval] summary:   {md_path}", flush=True)
    score = (OUTPUTS.get("assessment") or {}).get("vulnerability_score")
    print(
        f"[eval] outcome={outcome} | stages={len(STAGES)} "
        f"| total={telemetry['overall_wall_time_seconds']}s "
        f"| score={score}",
        flush=True,
    )

    if outcome == "completed" and not ARGS.keep:
        if _OWN_TEMP_DATA_DIR:
            shutil.rmtree(settings.data_dir, ignore_errors=True)
        if SYNTHETIC_WEIGHTS_DIR is not None:
            shutil.rmtree(SYNTHETIC_WEIGHTS_DIR, ignore_errors=True)
    else:
        print(f"[eval] artifacts kept at: {settings.data_dir}", flush=True)

    return 0 if outcome == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
