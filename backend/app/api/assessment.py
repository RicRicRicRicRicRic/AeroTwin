"""Assessment API: building profiles and seismic vulnerability assessments.

Endpoints under ``/api/assessment``:
* ``POST /profiles`` / ``GET /profiles`` / ``GET /profiles/{id}`` — building metadata.
* ``POST /calculate`` — queue a seismic assessment (HTTP 202, background task).
* ``GET /assessments`` / ``GET /assessments/{id}`` — assessment status & results.

All DB work runs in the Starlette threadpool so the event loop stays free.
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, status
from starlette.concurrency import run_in_threadpool

from ..core.database import get_job_record
from ..schemas.assessment import (
    AssessmentResponse,
    BuildingProfileCreate,
    BuildingProfileResponse,
    SeismicCalculationRequest,
)
from ..services import seismic_calculator
from ..services.seismic_calculator import (
    ProfileAlreadyExistsError,
    ProfileNotFoundError,
    run_seismic_assessment_job,
)

router = APIRouter(prefix="/api/assessment", tags=["assessment"])


@router.post(
    "/profiles",
    response_model=BuildingProfileResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a building profile",
)
async def create_profile(payload: BuildingProfileCreate) -> BuildingProfileResponse:
    """Register a building profile (unique name) used as assessment input."""
    try:
        return await run_in_threadpool(
            seismic_calculator.create_building_profile, payload
        )
    except ProfileAlreadyExistsError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc


@router.get(
    "/profiles",
    response_model=list[BuildingProfileResponse],
    summary="List building profiles",
)
async def list_profiles() -> list[BuildingProfileResponse]:
    """Return every registered building profile, newest first."""
    return await run_in_threadpool(seismic_calculator.list_building_profiles)


@router.get(
    "/profiles/{profile_id}",
    response_model=BuildingProfileResponse,
    summary="Get a building profile",
)
async def get_profile(profile_id: str) -> BuildingProfileResponse:
    """Return one building profile or HTTP 404."""
    profile = await run_in_threadpool(seismic_calculator.get_building_profile, profile_id)
    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown building profile: {profile_id}",
        )
    return profile


@router.post(
    "/calculate",
    response_model=AssessmentResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue a seismic vulnerability assessment",
)
async def calculate_assessment(
    request: SeismicCalculationRequest,
    background_tasks: BackgroundTasks,
) -> AssessmentResponse:
    """Score a building profile against mapped defect metrics (background task).

    Raises HTTP 404 for unknown profile/job references before anything is
    persisted, then returns the pending assessment immediately.
    """
    try:
        profile = await run_in_threadpool(
            seismic_calculator.get_profile_or_raise, request.profile_id
        )
    except ProfileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    for label, job_id in (
        ("crack_job_id", request.crack_job_id),
        ("element_job_id", request.element_job_id),
    ):
        if job_id is not None:
            row = await run_in_threadpool(get_job_record, job_id)
            if row is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Unknown {label}: {job_id}",
                )
    assessment_id = await run_in_threadpool(
        seismic_calculator.create_assessment, profile, request
    )
    background_tasks.add_task(run_seismic_assessment_job, assessment_id, request)
    assessment = await run_in_threadpool(seismic_calculator.get_assessment, assessment_id)
    if assessment is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Assessment {assessment_id} disappeared immediately after creation",
        )
    return assessment


@router.get(
    "/assessments",
    response_model=list[AssessmentResponse],
    summary="List assessments",
)
async def list_assessments(
    limit: int = Query(default=50, ge=1, le=200),
) -> list[AssessmentResponse]:
    """Return the most recent assessments, newest first."""
    return await run_in_threadpool(seismic_calculator.list_assessments, limit=limit)


@router.get(
    "/assessments/{assessment_id}",
    response_model=AssessmentResponse,
    summary="Get an assessment",
)
async def get_assessment(assessment_id: str) -> AssessmentResponse:
    """Return the status, score, and factor breakdown of an assessment."""
    assessment = await run_in_threadpool(seismic_calculator.get_assessment, assessment_id)
    if assessment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown assessment: {assessment_id}",
        )
    return assessment