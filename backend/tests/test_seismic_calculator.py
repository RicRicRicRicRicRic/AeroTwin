"""Unit and integration tests for Phase 4: seismic scoring & report generation.

Validates that the scoring engine is deterministic and repeatable, that the
documented weighted formula produces the expected numbers, that edge cases
(missing profile, unknown metadata, zero-defect buildings, absent element
damage) are handled cleanly, that metrics flow through ``aerotwin.db``, and
that reports land in ``data/outputs/``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.database import create_job_record, init_database, update_job_record
from app.main import app
from app.schemas.assessment import (
    BuildingProfileCreate,
    BuildingProfileResponse,
    CodeCompliance,
    ConstructionType,
    DefectSummary,
    SeismicAssessmentResult,
    SeismicCalculationRequest,
)
from app.schemas.processing import CrackFrameMetrics
from app.schemas.reports import ReportFormat, ReportGenerateRequest, ReportSummary
from app.services import report_generator, seismic_calculator
from app.services.defect_metrics import log_frame_metrics
from app.services.report_generator import run_report_generation
from app.services.seismic_calculator import (
    ProfileNotFoundError,
    calculate_vulnerability,
    classify_vulnerability,
    create_assessment,
    create_building_profile,
    derive_element_damage_states,
    get_assessment,
    get_profile_or_raise,
    run_seismic_assessment_job,
)


def make_profile(**overrides: object) -> BuildingProfileCreate:
    """Documented reference profile: 40 years, unreinforced masonry, 3 stories."""
    payload: dict[str, object] = {
        "name": "Municipal Hall",
        "structure_age_years": 40.0,
        "construction_type": ConstructionType.UNREINFORCED_MASONRY,
        "num_stories": 3,
        "code_compliance": CodeCompliance.PARTIALLY_COMPLIANT,
        "notes": "Reference profile for deterministic scoring tests.",
    }
    payload.update(overrides)
    return BuildingProfileCreate(**payload)  # type: ignore[arg-type]


def profile_response(**overrides: object) -> BuildingProfileResponse:
    """Build an in-memory profile record (no DB needed) for pure scoring tests."""
    payload = make_profile(**overrides)
    return BuildingProfileResponse(
        id="test-profile",
        created_at=datetime.now(timezone.utc),
        **payload.model_dump(),
    )


def make_defects(**overrides: object) -> DefectSummary:
    """Reference defect summary: 4 frames inspected, 2 with cracks."""
    payload: dict[str, object] = {
        "frames_analyzed": 4,
        "frames_with_cracks": 2,
        "mean_area_ratio": 1.0,
        "mean_width_px": 2.5,
        "mean_length_px": 80.0,
        "longest_crack_px": 90.0,
        "total_crack_pixels": 480,
    }
    payload.update(overrides)
    return DefectSummary(**payload)  # type: ignore[arg-type]


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return init_database(tmp_path / "aerotwin.db")


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


def seed_crack_job(
    db_path: Path,
    rows: list[CrackFrameMetrics] | None = None,
    *,
    output_dir: Path | None = None,
) -> str:
    """Create a crack-mapping job record with optional metric rows/output dir."""
    job_id = create_job_record(
        job_type="crack_mapping",
        video_filename="demo_video",
        params_json=json.dumps({"frames_path": "demo_video/run_test"}),
        db_path=db_path,
    )
    if rows:
        log_frame_metrics(job_id, rows, db_path)
    if output_dir is not None:
        update_job_record(job_id, {"output_dir": str(output_dir)}, db_path)
    return job_id


def crack_rows() -> list[CrackFrameMetrics]:
    """Two frames: one cracked (320 px, 80 px long, 4 px wide), one clean."""
    return [
        CrackFrameMetrics(
            frame_filename="frame_000000.jpg",
            crack_pixel_count=320,
            crack_area_ratio=4.0,
            crack_length_px=80.0,
            mean_width_px=4.0,
            component_count=2,
        ),
        CrackFrameMetrics(
            frame_filename="frame_000010.jpg",
            crack_pixel_count=0,
            crack_area_ratio=0.0,
            crack_length_px=0.0,
            mean_width_px=0.0,
            component_count=0,
        ),
    ]


# ---------------------------------------------------------------------------
# Scoring engine: determinism, documented formula, edge cases
# ---------------------------------------------------------------------------
def test_calculate_vulnerability_is_deterministic_and_repeatable() -> None:
    first = calculate_vulnerability(profile_response(), make_defects())
    second = calculate_vulnerability(profile_response(), make_defects())

    assert first.vulnerability_score == second.vulnerability_score
    assert first.classification == second.classification
    assert [f.model_dump() for f in first.factors] == [
        f.model_dump() for f in second.factors
    ]
    assert first.recommendations == second.recommendations
    assert 0.0 <= first.vulnerability_score <= 100.0


def test_expected_score_matches_documented_formula() -> None:
    result = calculate_vulnerability(profile_response(), make_defects())

    # Hand-computed from the documented weights (element factor unavailable → /0.85):
    #   age 0.8 x0.12 + construction 0.75 x0.14 + stories 2/9 x0.06
    #   + compliance 0.4 x0.08 + intensity 0.5 x0.30 + extent 0.5 x0.15
    #   = 0.471333 / 0.85 x 100 = 55.45
    assert result.vulnerability_score == pytest.approx(55.45, abs=0.01)
    assert result.classification == "Substantial"
    assert sum(factor.weight for factor in result.factors) == pytest.approx(1.0, abs=1e-9)
    assert "element_damage" not in {factor.name for factor in result.factors}
    assert result.recommendations[0].startswith("Commission a structural")


def test_zero_defect_building_scores_profile_factors_only() -> None:
    result = calculate_vulnerability(profile_response(), None)

    factors = {factor.name: factor for factor in result.factors}
    assert factors["crack_intensity"].value == 0.0
    assert factors["extent"].value == 0.0
    assert result.defect_summary.frames_analyzed == 0
    # 0.246333 / 0.85 x 100 = 28.98 (profile baseline only, no division bugs)
    assert result.vulnerability_score == pytest.approx(28.98, abs=0.01)
    assert result.classification == "Moderate"


def test_unknown_profile_metadata_uses_neutral_factors() -> None:
    profile = profile_response(
        structure_age_years=0.0,
        construction_type=ConstructionType.UNKNOWN,
        num_stories=1,
        code_compliance=CodeCompliance.UNKNOWN,
    )

    result = calculate_vulnerability(profile, make_defects())

    factors = {factor.name: factor.value for factor in result.factors}
    assert factors["age"] == 0.0
    assert factors["stories"] == 0.0
    assert factors["construction"] == 0.5  # neutral prior
    assert factors["compliance"] == 0.6  # neutral prior
    assert result.vulnerability_score > 0.0


def test_classification_tier_boundaries() -> None:
    assert classify_vulnerability(0.0) == "Low"
    assert classify_vulnerability(20.0) == "Low"
    assert classify_vulnerability(20.01) == "Moderate"
    assert classify_vulnerability(40.0) == "Moderate"
    assert classify_vulnerability(40.01) == "Substantial"
    assert classify_vulnerability(60.0) == "Substantial"
    assert classify_vulnerability(60.01) == "Severe"
    assert classify_vulnerability(80.0) == "Severe"
    assert classify_vulnerability(80.01) == "Critical"
    assert classify_vulnerability(100.0) == "Critical"


def test_element_damage_weight_is_redistributed_when_absent() -> None:
    without_element = calculate_vulnerability(profile_response(), make_defects())
    with_element = calculate_vulnerability(
        profile_response(),
        make_defects(
            element_damage_states={"intact": 1, "minor": 0, "moderate": 0, "severe": 3}
        ),
    )

    assert sum(f.weight for f in without_element.factors) == pytest.approx(1.0, abs=1e-9)
    assert sum(f.weight for f in with_element.factors) == pytest.approx(1.0, abs=1e-9)
    element_factor = {f.name: f for f in with_element.factors}["element_damage"]
    assert element_factor.value == pytest.approx(0.75, abs=1e-6)  # 3 severe of 4
    assert with_element.vulnerability_score > without_element.vulnerability_score


def test_missing_profile_raises_not_found(db_path: Path) -> None:
    with pytest.raises(ProfileNotFoundError) as excinfo:
        get_profile_or_raise("does-not-exist", db_path)

    assert "does-not-exist" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Element damage states from stored masks
# ---------------------------------------------------------------------------
def write_element_and_crack_masks(root: Path) -> tuple[Path, Path]:
    """Frame 0: moderate/intact/severe elements; frame 1: one minor element."""
    element_dir = root / "elements"
    crack_dir = root / "cracks"
    element_dir.mkdir(parents=True)
    crack_dir.mkdir(parents=True)

    frame0_elements = np.zeros((48, 64), dtype=np.uint8)
    frame0_elements[5:16, 5:16] = 1  # 121 px, 33 crack px → moderate
    frame0_elements[5:16, 25:36] = 2  # 121 px, no cracks → intact
    frame0_elements[20:31, 40:51] = 3  # 121 px, fully cracked → severe
    frame0_cracks = np.zeros((48, 64), dtype=np.uint8)
    frame0_cracks[5:16, 5:8] = 255
    frame0_cracks[20:31, 40:51] = 255

    frame1_elements = np.zeros((48, 64), dtype=np.uint8)
    frame1_elements[5:16, 5:16] = 1  # 121 px, 12 crack px → minor
    frame1_cracks = np.zeros((48, 64), dtype=np.uint8)
    frame1_cracks[5:9, 5:8] = 255

    assert cv2.imwrite(str(element_dir / "mask_000000.png"), frame0_elements)
    assert cv2.imwrite(str(crack_dir / "crack_mask_000000.png"), frame0_cracks)
    assert cv2.imwrite(str(element_dir / "mask_000010.png"), frame1_elements)
    assert cv2.imwrite(str(crack_dir / "crack_mask_000010.png"), frame1_cracks)
    return element_dir, crack_dir


def test_derive_element_damage_states_histogram(tmp_path: Path) -> None:
    element_dir, crack_dir = write_element_and_crack_masks(tmp_path / "run")

    states = derive_element_damage_states(element_dir, crack_dir)

    assert states == {"intact": 1, "minor": 1, "moderate": 1, "severe": 1}


def test_derive_element_damage_states_missing_dirs_returns_none(tmp_path: Path) -> None:
    element_dir, _ = write_element_and_crack_masks(tmp_path / "run")

    assert derive_element_damage_states(element_dir, tmp_path / "missing") is None


def test_derive_element_damage_states_skips_shape_mismatch(tmp_path: Path) -> None:
    element_dir = tmp_path / "elements"
    crack_dir = tmp_path / "cracks"
    element_dir.mkdir()
    crack_dir.mkdir()
    assert cv2.imwrite(str(element_dir / "mask_000000.png"), np.full((48, 64), 1, np.uint8))
    assert cv2.imwrite(str(crack_dir / "crack_mask_000000.png"), np.zeros((10, 10), np.uint8))

    assert derive_element_damage_states(element_dir, crack_dir) is None


# ---------------------------------------------------------------------------
# Assessment runner (DB writes + repeatability)
# ---------------------------------------------------------------------------
def test_run_seismic_assessment_job_writes_db_records(db_path: Path) -> None:
    profile = create_building_profile(make_profile(), db_path)
    crack_job_id = seed_crack_job(db_path, crack_rows())
    params = SeismicCalculationRequest(profile_id=profile.id, crack_job_id=crack_job_id)
    assessment_id = create_assessment(profile, params, db_path)

    run_seismic_assessment_job(assessment_id, params, db_path)
    first = get_assessment(assessment_id, db_path)

    assert first is not None
    assert first.status.value == "completed"
    assert first.vulnerability_score is not None
    assert first.classification is not None
    assert first.completed_at is not None
    assert first.error is None
    assert first.result is not None
    # Defect metrics flowed from frame_defect_metrics into the result.
    assert first.result.defect_summary.frames_analyzed == 2
    assert first.result.defect_summary.frames_with_cracks == 1
    # Area ratio averages over all inspected frames (4 % and 0 % → 2 %);
    # width/length average only over frames that actually contain cracks.
    assert first.result.defect_summary.mean_area_ratio == pytest.approx(2.0)
    assert first.result.defect_summary.mean_width_px == pytest.approx(4.0)
    assert first.result.defect_summary.mean_length_px == pytest.approx(80.0)
    assert first.result.vulnerability_score == first.vulnerability_score
    assert first.params["crack_job_id"] == crack_job_id

    # Re-running the same assessment reproduces the identical score.
    run_seismic_assessment_job(assessment_id, params, db_path)
    second = get_assessment(assessment_id, db_path)

    assert second is not None
    assert second.vulnerability_score == first.vulnerability_score
    assert second.classification == first.classification


def test_run_seismic_assessment_job_records_missing_profile(db_path: Path) -> None:
    profile = create_building_profile(make_profile(), db_path)
    params = SeismicCalculationRequest(profile_id="missing-profile")
    assessment_id = create_assessment(profile, params, db_path)

    run_seismic_assessment_job(assessment_id, params, db_path)

    assessment = get_assessment(assessment_id, db_path)
    assert assessment is not None
    assert assessment.status.value == "failed"
    assert assessment.error is not None and "Unknown building profile" in assessment.error


def test_run_seismic_assessment_job_records_unknown_job(db_path: Path) -> None:
    profile = create_building_profile(make_profile(), db_path)
    params = SeismicCalculationRequest(profile_id=profile.id, crack_job_id="no-such-job")
    assessment_id = create_assessment(profile, params, db_path)

    run_seismic_assessment_job(assessment_id, params, db_path)

    assessment = get_assessment(assessment_id, db_path)
    assert assessment is not None
    assert assessment.status.value == "failed"
    assert assessment.error is not None and "crack_job_id" in assessment.error


# ---------------------------------------------------------------------------
# Report generation (files in data/outputs/…, provenance in the DB)
# ---------------------------------------------------------------------------
def build_completed_assessment(
    db_path: Path,
    *,
    crack_output_dir: Path | None = None,
) -> tuple[str, str]:
    """Create profile + crack job + completed assessment; return (profile, assessment)."""
    profile = create_building_profile(make_profile(), db_path)
    crack_job_id = seed_crack_job(db_path, crack_rows(), output_dir=crack_output_dir)
    params = SeismicCalculationRequest(profile_id=profile.id, crack_job_id=crack_job_id)
    assessment_id = create_assessment(profile, params, db_path)
    run_seismic_assessment_job(assessment_id, params, db_path)
    return profile.id, assessment_id


def test_run_report_generation_writes_json_and_pdf(tmp_path: Path, db_path: Path) -> None:
    _, assessment_id = build_completed_assessment(db_path)
    request = ReportGenerateRequest(
        assessment_id=assessment_id, report_format=ReportFormat.BOTH
    )
    report_id = report_generator.create_report(assessment_id, ReportFormat.BOTH, db_path)

    run_report_generation(
        report_id,
        request,
        db_path=db_path,
        reports_root=tmp_path / "reports",
        assessments_root=tmp_path / "assessments",
    )

    report = report_generator.get_report(report_id, db_path)
    assert report is not None
    assert report.status.value == "completed"
    assert report.error is None
    json_path = Path(report.json_path or "")
    pdf_path = Path(report.pdf_path or "")
    findings_path = Path(report.assessment_json_path or "")
    assert json_path.is_file() and pdf_path.is_file() and findings_path.is_file()
    assert pdf_path.read_bytes()[:5] == b"%PDF-"  # real PDF magic bytes
    assert report.size_bytes is not None and report.size_bytes > 0

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    summary = ReportSummary.model_validate(payload)
    assert summary.assessment_id == assessment_id
    assert summary.report_id == report_id
    assert summary.result.vulnerability_score > 0.0
    assert summary.building.name == "Municipal Hall"

    findings = json.loads(findings_path.read_text(encoding="utf-8"))
    assert findings["assessment"]["assessment_id"] == assessment_id
    assert findings["result"]["vulnerability_score"] == summary.result.vulnerability_score


def test_run_report_generation_json_only(tmp_path: Path, db_path: Path) -> None:
    _, assessment_id = build_completed_assessment(db_path)
    request = ReportGenerateRequest(
        assessment_id=assessment_id, report_format=ReportFormat.JSON
    )
    report_id = report_generator.create_report(assessment_id, ReportFormat.JSON, db_path)

    run_report_generation(
        report_id,
        request,
        db_path=db_path,
        reports_root=tmp_path / "reports",
        assessments_root=tmp_path / "assessments",
    )

    report = report_generator.get_report(report_id, db_path)
    assert report is not None and report.status.value == "completed"
    assert report.json_path is not None and Path(report.json_path).is_file()
    assert report.pdf_path is None


def test_report_embeds_visual_defect_maps(tmp_path: Path, db_path: Path) -> None:
    crack_output = tmp_path / "crack_job_output"
    crack_output.mkdir()
    overlay = np.full((48, 64, 3), 120, dtype=np.uint8)
    assert cv2.imwrite(str(crack_output / "crack_overlay_000000.jpg"), overlay)
    _, assessment_id = build_completed_assessment(db_path, crack_output_dir=crack_output)
    request = ReportGenerateRequest(
        assessment_id=assessment_id, report_format=ReportFormat.BOTH
    )
    report_id = report_generator.create_report(assessment_id, ReportFormat.BOTH, db_path)

    run_report_generation(
        report_id,
        request,
        db_path=db_path,
        reports_root=tmp_path / "reports",
        assessments_root=tmp_path / "assessments",
    )

    report = report_generator.get_report(report_id, db_path)
    assert report is not None and report.status.value == "completed"
    payload = json.loads(Path(report.json_path or "").read_text(encoding="utf-8"))
    assert payload["visual_maps"] == [str(crack_output / "crack_overlay_000000.jpg")]
    assert payload["pipeline"]["crack_job_id"] is not None


def test_run_report_generation_records_unknown_assessment(
    tmp_path: Path, db_path: Path
) -> None:
    # A report row must reference an existing assessment (FK), so request a
    # different, nonexistent assessment id to exercise the guard.
    _, assessment_id = build_completed_assessment(db_path)
    report_id = report_generator.create_report(assessment_id, ReportFormat.JSON, db_path)
    request = ReportGenerateRequest(assessment_id="missing-assessment")

    run_report_generation(
        report_id,
        request,
        db_path=db_path,
        reports_root=tmp_path / "reports",
        assessments_root=tmp_path / "assessments",
    )

    report = report_generator.get_report(report_id, db_path)
    assert report is not None
    assert report.status.value == "failed"
    assert report.error is not None and "Unknown assessment" in report.error


def test_run_report_generation_requires_completed_assessment(
    tmp_path: Path, db_path: Path
) -> None:
    profile = create_building_profile(make_profile(), db_path)
    params = SeismicCalculationRequest(profile_id=profile.id)
    pending_id = create_assessment(profile, params, db_path)  # never executed
    request = ReportGenerateRequest(assessment_id=pending_id, report_format=ReportFormat.JSON)
    report_id = report_generator.create_report(pending_id, ReportFormat.JSON, db_path)

    run_report_generation(
        report_id,
        request,
        db_path=db_path,
        reports_root=tmp_path / "reports",
        assessments_root=tmp_path / "assessments",
    )

    report = report_generator.get_report(report_id, db_path)
    assert report is not None
    assert report.status.value == "failed"
    assert report.error is not None and "completed assessment is required" in report.error


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
def test_api_profile_endpoints(client: TestClient) -> None:
    payload = make_profile(name="API Profile Hall").model_dump(mode="json")

    created = client.post("/api/assessment/profiles", json=payload)

    assert created.status_code == 201
    body = created.json()
    assert body["name"] == "API Profile Hall"
    assert body["construction_type"] == "unreinforced_masonry"
    assert client.post("/api/assessment/profiles", json=payload).status_code == 409

    listing = client.get("/api/assessment/profiles")
    assert listing.status_code == 200
    assert any(item["id"] == body["id"] for item in listing.json())

    detail = client.get(f"/api/assessment/profiles/{body['id']}")
    assert detail.status_code == 200
    assert detail.json()["code_compliance"] == "partially_compliant"

    assert client.get("/api/assessment/profiles/unknown-profile").status_code == 404


def test_api_calculate_unknown_profile_returns_404(client: TestClient) -> None:
    response = client.post("/api/assessment/calculate", json={"profile_id": "nope"})

    assert response.status_code == 404


def test_api_calculate_unknown_job_returns_404(client: TestClient) -> None:
    created = client.post(
        "/api/assessment/profiles",
        json=make_profile(name="Unknown Job Building").model_dump(mode="json"),
    )
    assert created.status_code == 201

    response = client.post(
        "/api/assessment/calculate",
        json={"profile_id": created.json()["id"], "crack_job_id": "no-such-job"},
    )

    assert response.status_code == 404
    assert "no-such-job" in response.json()["detail"]


def test_api_assessment_and_report_workflow(client: TestClient) -> None:
    created = client.post(
        "/api/assessment/profiles",
        json=make_profile(name="Barangay Health Center").model_dump(mode="json"),
    )
    assert created.status_code == 201
    profile_id = created.json()["id"]
    crack_job_id = seed_crack_job(settings.database_path, crack_rows())

    queued = client.post(
        "/api/assessment/calculate",
        json={"profile_id": profile_id, "crack_job_id": crack_job_id},
    )
    assert queued.status_code == 202
    assessment_id = queued.json()["assessment_id"]

    # The TestClient runs background tasks before returning.
    final = client.get(f"/api/assessment/assessments/{assessment_id}")
    assert final.status_code == 200
    assessment = final.json()
    assert assessment["status"] == "completed"
    assert assessment["vulnerability_score"] > 0
    assert assessment["classification"]
    assert assessment["result"]["factors"]
    assert assessment["result"]["recommendations"]
    listing = client.get("/api/assessment/assessments")
    assert any(item["assessment_id"] == assessment_id for item in listing.json())

    generated = client.post(
        "/api/reports/generate",
        json={"assessment_id": assessment_id, "report_format": "both"},
    )
    assert generated.status_code == 202
    report_id = generated.json()["report_id"]

    report = client.get(f"/api/reports/{report_id}")
    assert report.status_code == 200
    report_body = report.json()
    assert report_body["status"] == "completed"
    assert Path(report_body["json_path"]).is_file()
    assert Path(report_body["pdf_path"]).is_file()
    assert Path(report_body["assessment_json_path"]).is_file()

    reports = client.get("/api/reports", params={"assessment_id": assessment_id})
    assert reports.status_code == 200
    assert [item["report_id"] for item in reports.json()] == [report_id]


def test_api_report_requires_completed_assessment(client: TestClient) -> None:
    created = client.post(
        "/api/assessment/profiles",
        json=make_profile(name="Pending Assessment Building").model_dump(mode="json"),
    )
    profile_id = created.json()["id"]
    profile = get_profile_or_raise(profile_id, settings.database_path)
    pending_id = create_assessment(
        profile, SeismicCalculationRequest(profile_id=profile_id), settings.database_path
    )

    response = client.post("/api/reports/generate", json={"assessment_id": pending_id})

    assert response.status_code == 409


def test_api_report_unknown_assessment_returns_404(client: TestClient) -> None:
    response = client.post("/api/reports/generate", json={"assessment_id": "nope"})

    assert response.status_code == 404
