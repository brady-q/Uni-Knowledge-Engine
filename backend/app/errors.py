"""Domain errors, deliberately free of any HTTP or FastAPI import.

Services raise these; ``app.main`` registers the handlers that turn them into
responses. That split is what keeps a service callable from a background task
or a script without dragging a web framework in, and keeps the choice of status
code in exactly one place instead of a try/except in every route handler.

Health checks return failures as values instead (see ``app.services.health``),
because "the database is down" is a report the endpoint must be able to render,
not an error that should abort the request.
"""

from __future__ import annotations


class DomainError(Exception):
    """Base class for expected, client-facing failures.

    ``detail`` is safe to return over HTTP: it is written for the caller, and
    never carries driver messages or SQL.
    """

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class NotFoundError(DomainError):
    """A referenced entity does not exist. Mapped to 404."""


class ConflictError(DomainError):
    """The request collides with existing state. Mapped to 409."""
