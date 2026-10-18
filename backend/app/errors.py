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


class PayloadTooLargeError(DomainError):
    """The uploaded body exceeds the configured limit. Mapped to 413."""


class UnsupportedMediaError(DomainError):
    """The uploaded bytes are not a file this API can ingest. Mapped to 415."""


class IngestionError(DomainError):
    """A document could not be turned into chunks.

    Unlike the errors above this one is never raised inside a request: ingestion
    runs after the response has been sent, so there is no status code to map it
    to. It is caught by the ingestion service, which records ``detail`` on
    ``documents.error`` - which is why the message has to be written for the
    person who uploaded the file rather than for a log reader.
    """


class EmbeddingError(DomainError):
    """The embedding model could not be used. Mapped to 503.

    Distinct from the errors above because it is not the caller's fault and not
    permanent. The model is several hundred megabytes and is loaded in a worker
    thread, so a request can legitimately arrive before it is ready, or after a
    load failed on a machine with no network. "Try again" is the honest answer;
    a 500 would suggest a bug, and a 4xx would blame the client.

    On the ingestion path there is no status code to return, so the pipeline
    translates this into an ``IngestionError`` recorded on the document row.
    """


class GenerationError(DomainError):
    """The answer model could not be reached or used. Mapped to 503.

    Covers a missing or rejected API key, a rate limit, an overloaded or
    unreachable service and a timeout. They differ in whether retrying helps,
    which is why the *detail* differs, but they share the only thing a status
    code can express: this server could not produce an answer, and the request
    itself was fine.

    Deliberately a sibling of ``EmbeddingError`` rather than a subclass. Both
    are 503, but one means "retrieval is unavailable" and the other "generation
    is unavailable", and a caller that wants to distinguish them - or a future
    handler that maps one differently - should not have to.
    """


class UnusableAnswerError(GenerationError):
    """The model responded, but with nothing that can be served. Mapped to 502.

    A refusal, an answer cut off at the token limit, or output that did not
    match the requested schema. 502 rather than 503 because the upstream was
    reachable and did reply; what came back is the problem. Retrying is not
    obviously useful, so the details say what to change instead.

    A subclass of ``GenerationError`` so that a caller which only cares that
    generation failed can catch the one type, while ``app.main`` still maps
    this to its own status - Starlette dispatches to the nearest registered
    handler, and both are registered.
    """
