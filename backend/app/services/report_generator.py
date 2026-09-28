"""Report generation service (Phase 4: Report Generation).

Aggregates the assessment findings, visual defect maps, and seismic
vulnerability score into structured outputs:

* ``data/outputs/assessments/`` — assessment findings JSON (profile + result)
* ``data/outputs/reports/``     — report summary JSON and/or PDF report

Unique filenames (``report_<id8>`` / ``assessment_<id8>_r<rep8>``) guarantee
outputs are never overwritten; raw inputs are never touched. All status
transitions are recorded in ``aerotwin.db`` (``reports`` table).
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

from ..core.config import settings
from ..core.database import get_connection, get_job_record
from ..schemas.assessment import AssessmentResponse, JobStatus
from ..schemas.reports import (
    ReportFormat,
    ReportGenerateRequest,
    ReportResponse,
    ReportSummary,
)
from .seismic_calculator import get_assessment, get_building_profile

logger = logging.getLogger("aerotwin.reports")

#: Maximum defect overlay images embedded into a PDF report.
MAX_VISUAL_MAPS: int = 4
#: PDF width for embedded defect maps.
VISUAL_MAP_WIDTH_MM: float = 150.0


class ReportGenerationError(Exception):
    """Base error for report generation (API maps to HTTP 4xx/5xx)."""


class ReportNotFoundError(ReportGenerationError):
    """The referenced assessment/report does not exist (HTTP 404)."""


class AssessmentNotReadyError(ReportGenerationError):
    """Assessment is not completed yet (HTTP 409)."""


class PdfUnavailableError(ReportGenerationError):
    """reportlab is not installed; PDF output cannot be produced."""


# ---------------------------------------------------------------------------
# Persistence: reports
# ---------------------------------------------------------------------------
def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


_REPORT_UPDATABLE_COLUMNS: frozenset[str] = frozenset(
    {
        "status",
        "json_path",
        "pdf_path",
        "assessment_json_path",
        "size_bytes",
        "error",
        "completed_at",
    }
)


def _row_to_report(row: Any) -> ReportResponse:
    return ReportResponse(
        report_id=row["id"],
        assessment_id=row["assessment_id"],
        status=JobStatus(row["status"]),
        report_format=ReportFormat(row["report_format"]),
        json_path=row["json_path"],
        pdf_path=row["pdf_path"],
        assessment_json_path=row["assessment_json_path"],
        size_bytes=row["size_bytes"],
        error=row["error"],
        created_at=datetime.fromisoformat(row["created_at"]),
        completed_at=(
            datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None
        ),
    )


def create_report(
    assessment_id: str,
    report_format: ReportFormat,
    db_path: Path | None = None,
) -> str:
    """Insert a ``pending`` report record and return its id."""
    report_id = uuid.uuid4().hex
    with get_connection(db_path) as connection:
        connection.execute(
            """
            INSERT INTO reports (
                id, assessment_id, status, report_format, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                report_id,
                assessment_id,
                JobStatus.PENDING.value,
                report_format.value,
                _utc_now().isoformat(),
            ),
        )
    logger.info("Created report %s for assessment %s", report_id, assessment_id)
    return report_id


def update_report(
    report_id: str,
    values: dict[str, Any],
    db_path: Path | None = None,
) -> None:
    """Persist selected report columns (whitelist-enforced)."""
    if not values:
        return
    unknown = set(values) - _REPORT_UPDATABLE_COLUMNS
    if unknown:
        raise ValueError(f"Cannot update report columns: {sorted(unknown)}")
    assignments = ", ".join(f"{column} = ?" for column in values)
    sql = f"UPDATE reports SET {assignments} WHERE id = ?"  # noqa: S608 (whitelisted identifiers)
    with get_connection(db_path) as connection:
        cursor = connection.execute(sql, (*values.values(), report_id))
        if cursor.rowcount == 0:
            raise KeyError(f"Unknown report: {report_id}")


def get_report(report_id: str, db_path: Path | None = None) -> ReportResponse | None:
    """Fetch a report record by id, or ``None``."""
    with get_connection(db_path) as connection:
        row = connection.execute(
            "SELECT * FROM reports WHERE id = ?", (report_id,)
        ).fetchone()
    return _row_to_report(row) if row is not None else None


def list_reports(
    limit: int = 50,
    assessment_id: str | None = None,
    db_path: Path | None = None,
) -> list[ReportResponse]:
    """Return recent reports (optionally for one assessment), newest first."""
    with get_connection(db_path) as connection:
        if assessment_id is not None:
            rows = connection.execute(
                """
                SELECT * FROM reports WHERE assessment_id = ?
                ORDER BY created_at DESC LIMIT ?
                """,
                (assessment_id, limit),
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT * FROM reports ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
    return [_row_to_report(row) for row in rows]


# ---------------------------------------------------------------------------
# Report content assembly
# ---------------------------------------------------------------------------
def _collect_visual_maps(
    assessment: AssessmentResponse,
    db_path: Path | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Locate defect overlay images and provenance for the source jobs."""
    crack_job_id = assessment.params.get("crack_job_id")
    element_job_id = assessment.params.get("element_job_id")
    pipeline: dict[str, Any] = {
        "crack_job_id": crack_job_id,
        "element_job_id": element_job_id,
        "crack_output_dir": None,
    }
    visual_maps: list[str] = []
    if crack_job_id:
        job_row = get_job_record(str(crack_job_id), db_path)
        output_dir = job_row["output_dir"] if job_row else None
        if output_dir:
            crack_output = Path(output_dir)
            pipeline["crack_output_dir"] = str(crack_output)
            if crack_output.is_dir():
                visual_maps = [
                    str(path)
                    for path in sorted(crack_output.glob("crack_overlay_*.jpg"))[
                        :MAX_VISUAL_MAPS
                    ]
                ]
    return visual_maps, pipeline


def build_report_summary(
    report_id: str,
    assessment: AssessmentResponse,
    db_path: Path | None = None,
) -> ReportSummary:
    """Assemble the structured report summary from a completed assessment.

    Raises:
        ReportGenerationError: Profile or result missing for the assessment.
    """
    profile = get_building_profile(assessment.profile_id, db_path)
    if profile is None:
        raise ReportGenerationError(
            f"Building profile {assessment.profile_id} not found for "
            f"assessment {assessment.assessment_id}"
        )
    if assessment.result is None:
        raise ReportGenerationError(
            f"Assessment {assessment.assessment_id} has no stored result"
        )
    visual_maps, pipeline = _collect_visual_maps(assessment, db_path)
    return ReportSummary(
        report_id=report_id,
        assessment_id=assessment.assessment_id,
        generated_at=_utc_now(),
        application=settings.app_name,
        application_version=settings.app_version,
        building=profile,
        result=assessment.result,
        visual_maps=visual_maps,
        pipeline=pipeline,
    )


# ---------------------------------------------------------------------------
# PDF rendering (reportlab; imported lazily so JSON reports work without it)
# ---------------------------------------------------------------------------
def _styled_table(rows: list[list[str]]) -> Any:
    """Build a grid-styled report table from string cells."""
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle

    table = Table(rows, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table


def _render_pdf(summary: ReportSummary, output_path: Path) -> None:
    """Render the inspection report as a PDF (A4) with defect-map thumbnails.

    Raises:
        PdfUnavailableError: reportlab is not installed.
        ReportGenerationError: A referenced defect map cannot be decoded.
    """
    try:
        from xml.sax.saxutils import escape

        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            Image,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
        )
    except ImportError as exc:
        raise PdfUnavailableError(
            "reportlab is not installed; PDF reports are unavailable "
            "(install with: pip install reportlab)."
        ) from exc

    styles = getSampleStyleSheet()
    story: list[Any] = []

    story.append(
        Paragraph(f"{escape(summary.application)} &ndash; Inspection Report", styles["Title"])
    )
    story.append(
        Paragraph(
            f"Generated {summary.generated_at.isoformat()} &nbsp;|&nbsp; "
            f"Report {escape(summary.report_id[:8])} &nbsp;|&nbsp; "
            f"Assessment {escape(summary.assessment_id[:8])}",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 8))

    building = summary.building
    profile_rows = [
        ["Field", "Value"],
        ["Building", building.name],
        ["Age (years)", f"{building.structure_age_years:g}"],
        ["Construction type", building.construction_type.value],
        ["Stories", str(building.num_stories)],
        ["Code compliance", building.code_compliance.value],
        ["Notes", building.notes or "-"],
    ]
    story.append(Paragraph("Building Profile", styles["Heading2"]))
    story.append(_styled_table(profile_rows))
    story.append(Spacer(1, 8))

    result = summary.result
    story.append(Paragraph("Seismic Vulnerability Assessment", styles["Heading2"]))
    story.append(
        Paragraph(
            f"<b>Score: {result.vulnerability_score:.2f} / 100 &mdash; "
            f"{escape(result.classification)}</b>",
            styles["Heading3"],
        )
    )
    factor_rows = [
        ["Factor", "Value", "Weight", "Contribution"],
        *[
            [
                factor.name,
                f"{factor.value:.4f}",
                f"{factor.weight:.4f}",
                f"{factor.contribution:.4f}",
            ]
            for factor in result.factors
        ],
    ]
    story.append(_styled_table(factor_rows))
    story.append(Spacer(1, 8))

    defect = result.defect_summary
    defect_rows = [
        ["Defect metric", "Value"],
        ["Frames analyzed", str(defect.frames_analyzed)],
        ["Frames with cracks", str(defect.frames_with_cracks)],
        ["Mean crack area ratio (%)", f"{defect.mean_area_ratio:.4f}"],
        ["Mean crack width (px)", f"{defect.mean_width_px:.4f}"],
        ["Mean crack length (px)", f"{defect.mean_length_px:.2f}"],
        ["Longest crack (px)", f"{defect.longest_crack_px:.2f}"],
        ["Total crack pixels", str(defect.total_crack_pixels)],
    ]
    if defect.element_damage_states:
        states = ", ".join(
            f"{name}: {count}" for name, count in defect.element_damage_states.items()
        )
        defect_rows.append(["Element damage states", states])
    story.append(Paragraph("Defect Summary", styles["Heading2"]))
    story.append(_styled_table(defect_rows))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Recommendations", styles["Heading2"]))
    for recommendation in result.recommendations:
        story.append(Paragraph(f"&bull; {escape(recommendation)}", styles["Normal"]))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Visual Defect Maps", styles["Heading2"]))
    maps_embedded = 0
    for map_path in summary.visual_maps:
        image_path = Path(map_path)
        if not image_path.is_file():
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ReportGenerationError(f"Unreadable defect map image: {image_path}")
        height, width = image.shape[:2]
        scaled_height = VISUAL_MAP_WIDTH_MM * height / max(width, 1)
        story.append(
            Image(str(image_path), width=VISUAL_MAP_WIDTH_MM * mm, height=scaled_height * mm)
        )
        story.append(Paragraph(escape(f"Source: {image_path.name}"), styles["Italic"]))
        story.append(Spacer(1, 6))
        maps_embedded += 1
    if maps_embedded == 0:
        story.append(Paragraph("No defect map images were available.", styles["Normal"]))

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        title=f"AeroTwin Inspection Report {summary.report_id[:8]}",
        author=summary.application,
    )
    document.build(story)


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------
def write_report_json(summary: ReportSummary, output_path: Path) -> None:
    """Write the structured report summary JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")


def write_assessment_findings(
    assessment: AssessmentResponse,
    summary: ReportSummary,
    output_path: Path,
) -> None:
    """Write the assessment findings JSON (assessment record + profile + result)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "assessment": assessment.model_dump(mode="json"),
        "building": summary.building.model_dump(mode="json"),
        "result": summary.result.model_dump(mode="json"),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Background report runner
# ---------------------------------------------------------------------------
def run_report_generation(
    report_id: str,
    request: ReportGenerateRequest,
    *,
    db_path: Path | None = None,
    reports_root: Path | None = None,
    assessments_root: Path | None = None,
) -> None:
    """Background task: compile inspection report files for an assessment.

    Writes to ``data/outputs/reports/`` and ``data/outputs/assessments/`` using
    unique names (existing outputs are never overwritten) and records status +
    file provenance in ``aerotwin.db``. Never raises — partial files are
    removed on failure and the error is stored on the report row.
    """
    update_report(report_id, {"status": JobStatus.RUNNING.value}, db_path)
    timer = time.perf_counter()
    written: list[Path] = []
    try:
        assessment = get_assessment(request.assessment_id, db_path)
        if assessment is None:
            raise ReportNotFoundError(f"Unknown assessment: {request.assessment_id}")
        if assessment.status != JobStatus.COMPLETED or assessment.result is None:
            raise AssessmentNotReadyError(
                f"Assessment {request.assessment_id} is {assessment.status.value}; "
                "a completed assessment is required to generate a report."
            )
        summary = build_report_summary(report_id, assessment, db_path)
        reports_dir = (
            reports_root if reports_root is not None else settings.outputs_dir / "reports"
        )
        assessments_dir = (
            assessments_root
            if assessments_root is not None
            else settings.outputs_dir / "assessments"
        )
        short_report = report_id[:8]
        short_assessment = assessment.assessment_id[:8]

        wants_json = request.report_format in (ReportFormat.JSON, ReportFormat.BOTH)
        wants_pdf = request.report_format in (ReportFormat.PDF, ReportFormat.BOTH)

        json_path: str | None = None
        pdf_path: str | None = None
        if wants_json:
            target = reports_dir / f"report_{short_report}.json"
            write_report_json(summary, target)
            written.append(target)
            json_path = str(target)

        findings_path = assessments_dir / f"assessment_{short_assessment}_r{short_report}.json"
        write_assessment_findings(assessment, summary, findings_path)
        written.append(findings_path)

        if wants_pdf:
            target = reports_dir / f"report_{short_report}.pdf"
            try:
                _render_pdf(summary, target)
            except PdfUnavailableError:
                if request.report_format is ReportFormat.PDF:
                    raise
                logger.warning(
                    "reportlab unavailable; report %s generated as JSON only", report_id
                )
            else:
                written.append(target)
                pdf_path = str(target)

        size_bytes = sum(path.stat().st_size for path in written)
        processing_time = time.perf_counter() - timer
        update_report(
            report_id,
            {
                "status": JobStatus.COMPLETED.value,
                "json_path": json_path,
                "pdf_path": pdf_path,
                "assessment_json_path": str(findings_path),
                "size_bytes": size_bytes,
                "completed_at": _utc_now().isoformat(),
            },
            db_path,
        )
        logger.info(
            "Report %s finished: %d file(s), %d bytes in %.3fs",
            report_id,
            len(written),
            size_bytes,
            processing_time,
        )
    except Exception as exc:  # noqa: BLE001 — job boundary: everything lands on the row
        for path in written:
            path.unlink(missing_ok=True)
        update_report(
            report_id,
            {
                "status": JobStatus.FAILED.value,
                "error": str(exc),
                "completed_at": _utc_now().isoformat(),
            },
            db_path,
        )
        logger.exception("Report %s failed", report_id)