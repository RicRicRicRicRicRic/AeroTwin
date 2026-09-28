"""Reports API: structural inspection report generation.

Endpoints under ``/api/reports``:
* ``POST /generate`` — compile JSON/PDF report files for a completed
  assessment (HTTP 202, background task; 404 unknown assessment, 409 if the
  assessment is not finished).
* ``GET ""`` / ``GET /{report_id}`` — report records with file paths and sizes.
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, status
from starlette.concurrency import run_in_threadpool

from ..schemas.assessment import JobStatus
from ..schemas.reports import ReportGenerateRequest, ReportResponse
from ..services import report_generator
from ..services.report_generator import run_report_generation

router = APIRouter(prefix="/api/reports", tags=["reports"])


@router.post(
    "/generate",
    response_model=ReportResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue inspection report generation",
)
async def generate_report(
    request: ReportGenerateRequest,
    background_tasks: BackgroundTasks,
) -> ReportResponse:
    """Compile report files for a completed assessment (background task)."""
    assessment = await run_in_threadpool(
        report_generator.get_assessment, request.assessment_id
    )
    if assessment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown assessment: {request.assessment_id}",
        )
    if assessment.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Assessment {request.assessment_id} is {assessment.status.value}; "
                "a completed assessment is required to generate a report."
            ),
        )
    report_id = await run_in_threadpool(
        report_generator.create_report, request.assessment_id, request.report_format
    )
    background_tasks.add_task(run_report_generation, report_id, request)
    report = await run_in_threadpool(report_generator.get_report, report_id)
    if report is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Report {report_id} disappeared immediately after creation",
        )
    return report


@router.get("", response_model=list[ReportResponse], summary="List reports")
async def list_reports(
    assessment_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
) -> list[ReportResponse]:
    """Return recent reports, optionally filtered to one assessment."""
    return await run_in_threadpool(
        report_generator.list_reports, limit=limit, assessment_id=assessment_id
    )


@router.get("/{report_id}", response_model=ReportResponse, summary="Get a report record")
async def get_report(report_id: str) -> ReportResponse:
    """Return one report record (including generated file paths) or HTTP 404."""
    report = await run_in_threadpool(report_generator.get_report, report_id)
    if report is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown report: {report_id}",
        )
    return report