"""Course CRUD.

Everything HTTP-shaped lives here and nowhere else: query-parameter bounds,
status codes, the ``Location`` header. The service is reached through a
dependency rather than constructed inline, so a test can swap in a fake and
exercise this contract without a database - the same trick
``get_readiness_report`` uses in ``app.routers.health``.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Path, Query, Request, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.schemas import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    CourseCreate,
    CoursePage,
    CourseRead,
    CourseUpdate,
)
from app.services.courses import CourseService

router = APIRouter(prefix="/courses", tags=["courses"])

# Documented in OpenAPI so /docs lists the failures a client must handle; the
# statuses themselves come from the handlers registered in app.main.
OpenApiResponses = dict[int | str, dict[str, Any]]

NOT_FOUND_RESPONSE: OpenApiResponses = {
    status.HTTP_404_NOT_FOUND: {"description": "No course with that id"}
}
CONFLICT_RESPONSE: OpenApiResponses = {
    status.HTTP_409_CONFLICT: {"description": "Course code already in use"}
}


def get_course_service(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> CourseService:
    return CourseService(session)


ServiceDep = Annotated[CourseService, Depends(get_course_service)]
CourseId = Annotated[uuid.UUID, Path(description="Course identifier")]


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    summary="Create a course",
    responses=CONFLICT_RESPONSE,
)
async def create_course(
    payload: CourseCreate,
    service: ServiceDep,
    request: Request,
    response: Response,
) -> CourseRead:
    course = await service.create(payload)
    # Where the thing that was just created can be fetched, which is what a 201
    # is required to say. url_for keeps it in step with the route below instead
    # of hard-coding the version prefix.
    response.headers["Location"] = str(request.url_for("get_course", course_id=course.id))
    return CourseRead.model_validate(course)


@router.get("", summary="List courses")
async def list_courses(
    service: ServiceDep,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = DEFAULT_PAGE_SIZE,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> CoursePage:
    courses, total = await service.list(limit=limit, offset=offset)
    return CoursePage(
        items=[CourseRead.model_validate(course) for course in courses],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{course_id}", summary="Fetch a course", responses=NOT_FOUND_RESPONSE)
async def get_course(course_id: CourseId, service: ServiceDep) -> CourseRead:
    return CourseRead.model_validate(await service.get(course_id))


@router.patch(
    "/{course_id}",
    summary="Update a course",
    responses={**NOT_FOUND_RESPONSE, **CONFLICT_RESPONSE},
)
async def update_course(
    course_id: CourseId, payload: CourseUpdate, service: ServiceDep
) -> CourseRead:
    return CourseRead.model_validate(await service.update(course_id, payload))


@router.delete(
    "/{course_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a course",
    responses=NOT_FOUND_RESPONSE,
)
async def delete_course(course_id: CourseId, service: ServiceDep) -> None:
    """Deleting a course also deletes its documents and chunks."""
    await service.delete(course_id)
