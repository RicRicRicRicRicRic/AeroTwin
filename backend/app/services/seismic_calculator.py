"""Seismic vulnerability scoring engine (Phase 4: Structural Assessment).

Combines **building profile metadata** (age, construction type, stories,
seismic-code compliance — stored in ``aerotwin.db``) with **mapped defect
metrics** (crack area ratios / widths / extents from ``frame_defect_metrics``
and element damage states derived by comparing element masks against crack
masks) into a standardised 0-100 vulnerability score and classification tier.

Design guarantees:
* **Deterministic** — a pure lookup/weighting model with fixed ordering and
  final rounding; identical inputs always produce identical scores.
* **Auditable** — every factor's value, weight, and contribution is returned
  and persisted so the thesis results chapter can reproduce each score.
* **Resilient** — unknown construction/compliance values map to neutral
  factors; a missing element-damage input redistributes its weight across
  the remaining factors instead of failing.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..core.database import get_connection, get_job_record
from ..schemas.assessment import (
    AssessmentResponse,
    BuildingProfileCreate,
    BuildingProfileResponse,
    CodeCompliance,
    ConstructionType,
    DefectSummary,
    FactorContribution,
    JobStatus,
    SeismicAssessmentResult,
    SeismicCalculationRequest,
)
from .defect_metrics import fetch_job_metrics, summarize_crack_metrics

logger = logging.getLogger("aerotwin.seismic")

# ---------------------------------------------------------------------------
# Scoring model constants (documented, fixed, deterministic)
# ---------------------------------------------------------------------------
#: Base weights of the six-plus-one factors (they sum to 1.0).
PROFILE_WEIGHTS: dict[str, float] = {
    "age": 0.12,
    "construction": 0.14,
    "stories": 0.06,
    "compliance": 0.08,
}
DEFECT_WEIGHTS: dict[str, float] = {
    "crack_intensity": 0.30,
    "extent": 0.15,
    "element_damage": 0.15,
}
#: Weight dropped (and redistributed) when element damage states are unavailable.
ELEMENT_WEIGHT: float = 0.15

#: Normalisation endpoints: value at which a factor reaches 1.0.
AGE_FULL_YEARS: float = 50.0
AREA_RATIO_FULL_PERCENT: float = 2.0  # mean crack area ratio >= 2 % → 1.0
WIDTH_FULL_PX: float = 5.0  # mean crack width >= 5 px → 1.0
STORIES_FULL: float = 10.0  # (stories - 1) / 9 → 1.0 at 10 stories

#: Lookup tables (documented vulnerability priors per construction system).
CONSTRUCTION_FACTORS: dict[ConstructionType, float] = {
    ConstructionType.RC_FRAME: 0.15,
    ConstructionType.STEEL: 0.25,
    ConstructionType.TIMBER: 0.55,
    ConstructionType.REINFORCED_MASONRY: 0.45,
    ConstructionType.UNREINFORCED_MASONRY: 0.75,
    ConstructionType.ADOBE: 0.95,
    ConstructionType.UNKNOWN: 0.50,  # neutral prior
}
COMPLIANCE_FACTORS: dict[CodeCompliance, float] = {
    CodeCompliance.COMPLIANT: 0.0,
    CodeCompliance.PARTIALLY_COMPLIANT: 0.4,
    CodeCompliance.NON_COMPLIANT: 0.8,
    CodeCompliance.UNKNOWN: 0.6,  # neutral prior
}

#: Classification tiers: (inclusive upper score bound, label).
CLASSIFICATION_TIERS: tuple[tuple[float, str], ...] = (
    (20.0, "Low"),
    (40.0, "Moderate"),
    (60.0, "Substantial"),
    (80.0, "Severe"),
)
DEFAULT_TIER: str = "Critical"

#: Element damage states: (exclusive upper crack density, state label).
ELEMENT_DAMAGE_THRESHOLDS: tuple[tuple[float, str], ...] = (
    (0.05, "intact"),
    (0.15, "minor"),
    (0.30, "moderate"),
)
ELEMENT_SEVERE_STATE: str = "severe"
#: Severity weights used to fold the state histogram into a 0-1 factor.
ELEMENT_STATE_SEVERITY: dict[str, int] = {
    "intact": 0,
    "minor": 1,
    "moderate": 2,
    "severe": 3,
}
#: Fixed output order of element damage state counts.
ELEMENT_STATE_ORDER: tuple[str, ...] = ("intact", "minor", "moderate", "severe")

#: Factor iteration order for the audit trail (stable across runs).
FACTOR_ORDER: tuple[str, ...] = (
    "age",
    "construction",
    "stories",
    "compliance",
    "crack_intensity",
    "extent",
    "element_damage",
)


class ProfileNotFoundError(Exception):
    """The requested building profile does not exist (API maps to HTTP 404)."""


class ProfileAlreadyExistsError(Exception):
    """A building profile with the same name exists (API maps to HTTP 409)."""


def classify_vulnerability(score: float) -> str:
    """Map a 0-100 score onto its standardised classification tier."""
    for upper_bound, label in CLASSIFICATION_TIERS:
        if score <= upper_bound:
            return label
    return DEFAULT_TIER


def _clamp01(value: float) -> float:
    """Clamp a factor value into the [0, 1] range."""
    return max(0.0, min(1.0, value))


def _element_damage_factor(states: dict[str, int]) -> float:
    """Fold a state histogram into 0-1: severity-weighted mean (3 = severe)."""
    total = sum(states.get(state, 0) for state in ELEMENT_STATE_ORDER)
    if total <= 0:
        return 0.0
    weighted = sum(
        ELEMENT_STATE_SEVERITY[state] * int(states.get(state, 0))
        for state in ELEMENT_STATE_ORDER
    )
    return weighted / (3.0 * total)


_BASE_RECOMMENDATIONS: dict[str, str] = {
    "Low": "No urgent action: re-inspect after the next seismic event or within 24 months.",
    "Moderate": "Schedule a detailed manual inspection within 12 months and monitor crack propagation.",
    "Substantial": "Commission a structural engineering evaluation within 6 months and install crack monitors.",
    "Severe": "Restrict occupancy of affected areas and obtain an urgent seismic retrofit assessment.",
    "Critical": "Immediately restrict occupancy and arrange an emergency structural intervention assessment.",
}


def _build_recommendations(
    classification: str,
    profile: BuildingProfileResponse,
    summary: DefectSummary,
) -> list[str]:
    """Rule-based, deterministic recommendations for the report."""
    recommendations: list[str] = [_BASE_RECOMMENDATIONS[classification]]
    compliance = CodeCompliance(profile.code_compliance)
    if compliance is CodeCompliance.NON_COMPLIANT:
        recommendations.append(
            "Design and implement a seismic retrofit to reach current code compliance."
        )
    elif compliance in (CodeCompliance.PARTIALLY_COMPLIANT, CodeCompliance.UNKNOWN):
        recommendations.append(
            "Verify seismic-code compliance status and document any retrofit history."
        )
    construction = ConstructionType(profile.construction_type)
    if construction in (ConstructionType.UNREINFORCED_MASONRY, ConstructionType.ADOBE):
        recommendations.append(
            f"Prioritise {construction.value} elements for confinement or jacketing before other works."
        )
    if profile.structure_age_years >= 40.0:
        recommendations.append(
            "Structure is 40+ years old: verify original drawings and assess material degradation."
        )
    if (
        summary.frames_analyzed > 0
        and summary.frames_with_cracks * 2 >= summary.frames_analyzed
    ):
        recommendations.append(
            "Cracks affect at least half of the inspected frames: measure crack widths "
            "in-situ (GSD-calibrated)."
        )
    states = summary.element_damage_states or {}
    if states.get("severe", 0) > 0:
        recommendations.append(
            "Detected elements show severe cracking: verify load paths without delay."
        )
    return recommendations


def calculate_vulnerability(
    profile: BuildingProfileResponse,
    defects: DefectSummary | None,
) -> SeismicAssessmentResult:
    """Compute the deterministic vulnerability score for a building.

    Score = 100 x sum(normalised factor x weight), rounded to 2 decimals.
    Weights always sum to 1.0: when element damage states are unavailable the
    element weight (0.15) is redistributed proportionally across the other
    factors, so zero-defect and profile-only assessments stay well-defined.
    """
    summary = defects if defects is not None else DefectSummary()

    values: dict[str, float] = {
        "age": _clamp01(max(profile.structure_age_years, 0.0) / AGE_FULL_YEARS),
        "construction": CONSTRUCTION_FACTORS[ConstructionType(profile.construction_type)],
        "stories": _clamp01((max(profile.num_stories, 1) - 1) / (STORIES_FULL - 1.0)),
        "compliance": COMPLIANCE_FACTORS[CodeCompliance(profile.code_compliance)],
        "crack_intensity": (
            _clamp01(summary.mean_area_ratio / AREA_RATIO_FULL_PERCENT)
            + _clamp01(summary.mean_width_px / WIDTH_FULL_PX)
        )
        / 2.0,
        "extent": (
            _clamp01(summary.frames_with_cracks / summary.frames_analyzed)
            if summary.frames_analyzed > 0
            else 0.0
        ),
    }

    base_weights: dict[str, float] = {**PROFILE_WEIGHTS, **DEFECT_WEIGHTS}
    if summary.element_damage_states is not None:
        values["element_damage"] = _element_damage_factor(summary.element_damage_states)
        weights = base_weights
    else:
        active = [name for name in FACTOR_ORDER if name != "element_damage"]
        active_total = sum(base_weights[name] for name in active)
        weights = {name: base_weights[name] / active_total for name in active}

    contributions = [
        FactorContribution(
            name=name,
            value=round(values[name], 6),
            weight=round(weights[name], 6),
            contribution=round(values[name] * weights[name], 6),
        )
        for name in FACTOR_ORDER
        if name in weights
    ]
    raw_score = sum(values[name] * weights[name] for name in FACTOR_ORDER if name in weights)
    score = round(100.0 * raw_score, 2)
    classification = classify_vulnerability(score)

    return SeismicAssessmentResult(
        vulnerability_score=score,
        classification=classification,
        factors=contributions,
        defect_summary=summary,
        recommendations=_build_recommendations(classification, profile, summary),
    )


# ---------------------------------------------------------------------------
# Persistence: building profiles
# ---------------------------------------------------------------------------
def _utc_now() -> datetime:
    """Timezone-aware current time (all persisted timestamps are UTC ISO-8601)."""
    return datetime.now(timezone.utc)


def _row_to_profile(row: Any) -> BuildingProfileResponse:
    return BuildingProfileResponse(
        id=row["id"],
        name=row["name"],
        structure_age_years=row["structure_age_years"],
        construction_type=ConstructionType(row["construction_type"]),
        num_stories=row["num_stories"],
        code_compliance=CodeCompliance(row["code_compliance"]),
        notes=row["notes"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def create_building_profile(
    payload: BuildingProfileCreate,
    db_path: Path | None = None,
) -> BuildingProfileResponse:
    """Persist a building profile; names are unique (duplicates raise)."""
    name = payload.name.strip()
    if not name:
        raise ValueError("Building profile name must not be empty.")
    with get_connection(db_path) as connection:
        existing = connection.execute(
            "SELECT id FROM building_profiles WHERE name = ?", (name,)
        ).fetchone()
        if existing is not None:
            raise ProfileAlreadyExistsError(f"Building profile already exists: {name!r}")
        profile_id = uuid.uuid4().hex
        created_at = _utc_now()
        connection.execute(
            """
            INSERT INTO building_profiles (
                id, name, structure_age_years, construction_type,
                num_stories, code_compliance, notes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                profile_id,
                name,
                payload.structure_age_years,
                payload.construction_type.value,
                payload.num_stories,
                payload.code_compliance.value,
                payload.notes,
                created_at.isoformat(),
            ),
        )
    logger.info("Created building profile %s (%s)", profile_id, name)
    return BuildingProfileResponse(
        id=profile_id,
        created_at=created_at,
        **payload.model_dump(exclude={"name"}),
        name=name,
    )


def get_building_profile(
    profile_id: str,
    db_path: Path | None = None,
) -> BuildingProfileResponse | None:
    """Fetch a profile by id, or ``None``."""
    with get_connection(db_path) as connection:
        row = connection.execute(
            "SELECT * FROM building_profiles WHERE id = ?", (profile_id,)
        ).fetchone()
    return _row_to_profile(row) if row is not None else None


def get_profile_or_raise(
    profile_id: str,
    db_path: Path | None = None,
) -> BuildingProfileResponse:
    """Fetch a profile or raise :class:`ProfileNotFoundError` (API maps to 404)."""
    profile = get_building_profile(profile_id, db_path)
    if profile is None:
        raise ProfileNotFoundError(f"Unknown building profile: {profile_id}")
    return profile


def list_building_profiles(db_path: Path | None = None) -> list[BuildingProfileResponse]:
    """Return all profiles, newest first."""
    with get_connection(db_path) as connection:
        rows = connection.execute(
            "SELECT * FROM building_profiles ORDER BY created_at DESC, name ASC"
        ).fetchall()
    return [_row_to_profile(row) for row in rows]


# ---------------------------------------------------------------------------
# Persistence: assessments
# ---------------------------------------------------------------------------
_ASSESSMENT_UPDATABLE_COLUMNS: frozenset[str] = frozenset(
    {
        "status",
        "result_json",
        "vulnerability_score",
        "classification",
        "error",
        "completed_at",
    }
)


def _row_to_assessment(row: Any) -> AssessmentResponse:
    result = (
        SeismicAssessmentResult.model_validate_json(row["result_json"])
        if row["result_json"]
        else None
    )
    return AssessmentResponse(
        assessment_id=row["id"],
        status=JobStatus(row["status"]),
        profile_id=row["profile_id"],
        building_name=row["building_name"],
        params=json.loads(row["params_json"]) if row["params_json"] else {},
        vulnerability_score=row["vulnerability_score"],
        classification=row["classification"],
        result=result,
        error=row["error"],
        created_at=datetime.fromisoformat(row["created_at"]),
        completed_at=(
            datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None
        ),
    )


def create_assessment(
    profile: BuildingProfileResponse,
    params: SeismicCalculationRequest,
    db_path: Path | None = None,
) -> str:
    """Insert a ``pending`` assessment and return its id."""
    assessment_id = uuid.uuid4().hex
    with get_connection(db_path) as connection:
        connection.execute(
            """
            INSERT INTO assessments (
                id, profile_id, building_name, status, params_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                assessment_id,
                profile.id,
                profile.name,
                JobStatus.PENDING.value,
                params.model_dump_json(),
                _utc_now().isoformat(),
            ),
        )
    logger.info("Created assessment %s for %s", assessment_id, profile.name)
    return assessment_id


def update_assessment(
    assessment_id: str,
    values: dict[str, Any],
    db_path: Path | None = None,
) -> None:
    """Persist selected assessment columns (whitelist-enforced).

    Raises:
        KeyError: Unknown assessment id. ValueError: Non-whitelisted column.
    """
    if not values:
        return
    unknown = set(values) - _ASSESSMENT_UPDATABLE_COLUMNS
    if unknown:
        raise ValueError(f"Cannot update assessment columns: {sorted(unknown)}")
    assignments = ", ".join(f"{column} = ?" for column in values)
    sql = f"UPDATE assessments SET {assignments} WHERE id = ?"  # noqa: S608 (whitelisted identifiers)
    with get_connection(db_path) as connection:
        cursor = connection.execute(sql, (*values.values(), assessment_id))
        if cursor.rowcount == 0:
            raise KeyError(f"Unknown assessment: {assessment_id}")


def get_assessment(
    assessment_id: str,
    db_path: Path | None = None,
) -> AssessmentResponse | None:
    """Fetch an assessment by id, or ``None``."""
    with get_connection(db_path) as connection:
        row = connection.execute(
            "SELECT * FROM assessments WHERE id = ?", (assessment_id,)
        ).fetchone()
    return _row_to_assessment(row) if row is not None else None


def list_assessments(
    limit: int = 50,
    db_path: Path | None = None,
) -> list[AssessmentResponse]:
    """Return the most recent assessments, newest first."""
    with get_connection(db_path) as connection:
        rows = connection.execute(
            "SELECT * FROM assessments ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [_row_to_assessment(row) for row in rows]


# ---------------------------------------------------------------------------
# Defect integration (frame metrics + element damage states)
# ---------------------------------------------------------------------------
def _state_for_density(density: float) -> str:
    """Map a crack-pixel density inside one element to its damage state."""
    for upper_bound, label in ELEMENT_DAMAGE_THRESHOLDS:
        if density < upper_bound:
            return label
    return ELEMENT_SEVERE_STATE


def derive_element_damage_states(
    element_output_dir: Path,
    crack_output_dir: Path,
) -> dict[str, int] | None:
    """Grade every detected element instance by crack density inside it.

    For each element mask (``mask_<frame>.png`` from an element-detection job)
    the matching crack mask (``crack_mask_<frame>.png`` from a crack-mapping
    job) is intersected with every connected component; the crack-pixel
    density selects the state (``<5 % intact, <15 % minor, <30 % moderate,
    else severe``). Frames missing from either run are skipped.

    Returns a state histogram, or ``None`` when nothing could be evaluated.
    """
    if not element_output_dir.is_dir() or not crack_output_dir.is_dir():
        return None
    counts: dict[str, int] = {state: 0 for state in ELEMENT_STATE_ORDER}
    evaluated = 0
    for element_path in sorted(element_output_dir.glob("mask_*.png")):
        frame_id = element_path.stem.removeprefix("mask_")
        crack_path = crack_output_dir / f"crack_mask_{frame_id}.png"
        elements = cv2.imread(str(element_path), cv2.IMREAD_UNCHANGED)
        if elements is None or not crack_path.is_file():
            continue
        cracks = cv2.imread(str(crack_path), cv2.IMREAD_UNCHANGED)
        if cracks is None:
            continue
        if elements.shape != cracks.shape:
            logger.warning(
                "Skipping frame %s: element/crack mask shapes disagree (%s vs %s)",
                frame_id,
                elements.shape,
                cracks.shape,
            )
            continue
        crack_bool = cracks > 0
        for class_index in (int(value) for value in np.unique(elements) if value > 0):
            class_mask = (elements == class_index).astype(np.uint8)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(
                class_mask, connectivity=8
            )
            for component in range(1, count):
                area = int(stats[component, cv2.CC_STAT_AREA])
                if area <= 0:
                    continue
                density = float(crack_bool[labels == component].mean())
                counts[_state_for_density(density)] += 1
                evaluated += 1
    if evaluated == 0:
        return None
    logger.info("Element damage states: %s (%d instances)", counts, evaluated)
    return counts


def aggregate_defect_metrics(
    crack_job_id: str | None,
    element_job_id: str | None,
    db_path: Path | None = None,
) -> DefectSummary:
    """Aggregate defect inputs for the scoring engine.

    Crack metrics come from ``frame_defect_metrics`` rows of the crack-mapping
    job (missing rows are treated as zero defects, with a warning); element
    damage states are derived by intersecting the two jobs' stored masks.
    """
    rows = fetch_job_metrics(crack_job_id, db_path) if crack_job_id else []
    if crack_job_id and not rows:
        logger.warning(
            "Crack job %s has no metric rows; treating as zero-defect.", crack_job_id
        )
    if rows:
        aggregate = summarize_crack_metrics(rows)
        defect_summary = DefectSummary(
            frames_analyzed=aggregate.frames_analyzed,
            frames_with_cracks=aggregate.frames_with_cracks,
            mean_area_ratio=aggregate.mean_area_ratio,
            mean_width_px=aggregate.mean_width_px,
            mean_length_px=aggregate.mean_crack_length_px,
            longest_crack_px=aggregate.longest_crack_px,
            total_crack_pixels=aggregate.total_crack_pixels,
        )
    else:
        defect_summary = DefectSummary()

    element_states: dict[str, int] | None = None
    if element_job_id and crack_job_id:
        element_row = get_job_record(element_job_id, db_path)
        crack_row = get_job_record(crack_job_id, db_path)
        element_out = Path(element_row["output_dir"]) if element_row and element_row["output_dir"] else None
        crack_out = Path(crack_row["output_dir"]) if crack_row and crack_row["output_dir"] else None
        if element_out is not None and crack_out is not None:
            element_states = derive_element_damage_states(element_out, crack_out)
        if element_states is None:
            logger.warning(
                "Element damage states unavailable (job outputs missing or empty); "
                "the element factor weight will be redistributed."
            )
    defect_summary.element_damage_states = element_states
    return defect_summary


# ---------------------------------------------------------------------------
# Background assessment runner
# ---------------------------------------------------------------------------
def run_seismic_assessment_job(
    assessment_id: str,
    params: SeismicCalculationRequest,
    db_path: Path | None = None,
) -> None:
    """Background task: compute the vulnerability score for an assessment.

    Never raises — failures are recorded on the assessment row so the API and
    frontend always have a structured error instead of a traceback. Inputs
    (profile + job ids) and the resulting score/factors are persisted as the
    assessment's telemetry record.
    """
    update_assessment(assessment_id, {"status": JobStatus.RUNNING.value}, db_path)
    timer = time.perf_counter()
    try:
        profile = get_profile_or_raise(params.profile_id, db_path)
        for job_label, job_id in (
            ("crack_job_id", params.crack_job_id),
            ("element_job_id", params.element_job_id),
        ):
            if job_id and get_job_record(job_id, db_path) is None:
                raise ValueError(f"Unknown {job_label}: {job_id}")
        defect_summary = aggregate_defect_metrics(
            params.crack_job_id, params.element_job_id, db_path
        )
        result = calculate_vulnerability(profile, defect_summary)
        processing_time = time.perf_counter() - timer
        update_assessment(
            assessment_id,
            {
                "status": JobStatus.COMPLETED.value,
                "result_json": result.model_dump_json(),
                "vulnerability_score": result.vulnerability_score,
                "classification": result.classification,
                "completed_at": _utc_now().isoformat(),
            },
            db_path,
        )
        logger.info(
            "Assessment %s finished: %.2f (%s) for %s in %.3fs",
            assessment_id,
            result.vulnerability_score,
            result.classification,
            profile.name,
            processing_time,
        )
    except Exception as exc:  # noqa: BLE001 — job boundary: everything lands on the row
        update_assessment(
            assessment_id,
            {
                "status": JobStatus.FAILED.value,
                "error": str(exc),
                "completed_at": _utc_now().isoformat(),
            },
            db_path,
        )
        logger.exception("Assessment %s failed", assessment_id)
