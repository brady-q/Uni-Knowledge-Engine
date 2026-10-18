"""The answer model, behind a narrow interface.

This is the only module that talks to Anthropic, and it lives at the top level
next to ``embedding.py`` and ``storage.py`` for the same reason those do: it is
infrastructure the services are *handed*, and it owns the FastAPI dependency
that hands it over. The rule that keeps FastAPI out applies to services,
repositories and models, not to the seams above them.

The interface is one method wide, and its signature is the whole design:

    async def generate(question: str, passages: Sequence[str]) -> AnswerDraft

``passages`` is a sequence of **strings**. Not chunks, not rows, not anything
carrying a filename or a page number - because a model cannot repeat a page
number it was never shown, and that is the mechanism the citation guarantee
rests on rather than an instruction the model is trusted to follow. Passage
``i`` is presented as ``[S{i+1}]`` and the model can refer to it only by that
opaque marker; ``app.services.answers`` maps the markers back onto the rows
they came from and reads the real metadata out of the database.

The other half of the contract is the response format. The model replies with
JSON matching ``DraftAnswer`` via ``output_config.format``, so there is no
preamble to strip and no "sure, here is your answer" to parse around. What it
cannot do is make the *markers* trustworthy: a schema constrains shape, not
content, and the model can still emit ``[S9]`` when it was shown six passages.
Dropping those is the resolver's job, and it is not optional.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import anthropic
from fastapi import Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.errors import GenerationError, UnusableAnswerError

logger = logging.getLogger(__name__)

type Effort = Literal["low", "medium", "high", "xhigh", "max"]

# Retrieval hands over half a dozen short passages and the answer is a
# paragraph, so the ceiling is here to bound a runaway generation rather than
# to fit the work. Adaptive thinking is billed against it too, which is why it
# is not tighter.
DEFAULT_MAX_TOKENS = 4096

# "low" because this is a small, well-specified task: read six passages,
# answer from them, mark what came from where. Effort buys depth on problems
# that need it, and this one is bounded by what the passages say - so the
# budget goes to a better retrieval, not to a longer deliberation. Raise it in
# the environment (ANSWER_EFFORT) if answers come back shallow.
DEFAULT_EFFORT: Effort = "low"

# Somebody is waiting on an HTTP response, so the SDK's default of two retries
# - three attempts, each up to the full timeout - is too patient here. One
# retry still covers a single transient 429 or 529.
MAX_RETRIES = 1

# --- what a caller is told when there is no answer ---------------------------
#
# All of these are written for whoever asked the question, not for a log
# reader. Where the honest advice is "try again", they say so; where it is
# "this server is misconfigured", they do not pretend otherwise.

NOT_CONFIGURED_DETAIL = "answering is not configured on this server: ANTHROPIC_API_KEY is not set"
REJECTED_DETAIL = "answering is not configured correctly on this server"
BUSY_DETAIL = "the answer service is busy; try again shortly"
FAILED_DETAIL = "the answer service could not be used; see the server logs"
REFUSED_DETAIL = "the model declined to answer this question"
TRUNCATED_DETAIL = "the answer ran past its length limit; try a narrower question"
MALFORMED_DETAIL = "the model returned an answer that could not be read"

# Markers are opaque on purpose: "S3" tells the model nothing about which
# document or page it is pointing at, so there is nothing for it to paraphrase
# into a citation of its own. The leading horizontal whitespace is part of the
# match so that deleting an invented marker leaves "as shown." rather than
# "as shown ."; a valid marker is substituted back unchanged, whitespace and
# all. Newlines are excluded so that removing a marker cannot join two lines.
MARKER_PATTERN = re.compile(r"[ \t]*\[S(?P<index>\d+)\]")


def marker(index: int) -> str:
    """The label passage ``index`` (1-based) is shown under."""
    return f"S{index}"


SYSTEM_PROMPT = """\
You answer questions about one university course, using only the passages \
supplied with each question.

Every passage is labelled with an opaque marker - [S1], [S2], and so on. Those \
markers are the only way you can refer to a passage.

Rules:

1. Use only the supplied passages. Treat anything you happen to know about \
this subject as unavailable: the passages are the course.
2. Put the marker of the passage a claim came from immediately after that \
claim, e.g. "The second assignment is due in week six [S2]." Every factual \
claim needs one. A claim resting on two passages takes both: "[S1][S4]".
3. Only the markers listed below the passages exist. Never write one that was \
not supplied - it will be deleted from your answer, leaving the claim it was \
meant to support standing on nothing.
4. Never write a page number, a file name, a section number or a date. You \
have not been shown any, so anything you wrote would be invented. The marker \
is the reference, and the citation is assembled around it afterwards.
5. Do not talk about the passages, the markers or where they came from. No \
"the provided context", no "Source 1 states", no "according to the materials". \
Answer the question; the markers do the referring.
6. If the passages do not contain what the question needs, set "answered" to \
false and say so in one plain sentence with no markers in it. Do not close the \
gap with general knowledge, and do not answer a nearby question instead.
7. At most a short paragraph, in the register of a teaching assistant who has \
read the material.\
"""


class DraftAnswer(BaseModel):
    """The only shape the model is allowed to reply in.

    Two fields, because the prose is the single citation channel: the markers
    are written inline where the claims are, and there is deliberately no
    second list of "sources used" that could disagree with them.

    ``answered`` is separate from "did it cite anything?" so that an honest
    abstention is distinguishable from a confident answer that simply forgot
    to cite. The answer service treats disagreement between the two as the
    abstention.
    """

    # A field the schema did not ask for is a signal the prompt and the schema
    # have drifted, and it should fail here rather than be quietly dropped.
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(
        description=(
            "The answer, with [S#] markers inline immediately after the claims they support."
        )
    )
    answered: bool = Field(
        description=(
            "False if the passages did not contain what the question needed, "
            "in which case the answer says so and cites nothing."
        )
    )


def _draft_schema() -> dict[str, Any]:
    """The schema as it goes over the wire.

    Derived from ``DraftAnswer`` rather than written out a second time: a
    schema that has drifted from the type it is validated into is a bug that
    only appears in production. ``extra="forbid"`` is what emits
    ``additionalProperties: false``, which structured outputs requires.

    What is stripped is the prose Pydantic generates for itself. The class
    docstring becomes a top-level ``description`` and every name becomes a
    ``title``, and all of it is sent to the model as part of the prompt - so
    the paragraphs above, written for whoever maintains this file, would end up
    being read as instructions. The field ``description``s stay, because those
    are deliberately addressed to the model.
    """
    schema = DraftAnswer.model_json_schema()
    schema.pop("description", None)
    schema.pop("title", None)
    for field in schema["properties"].values():
        field.pop("title", None)
    return schema


DRAFT_SCHEMA: dict[str, Any] = _draft_schema()


@dataclass(frozen=True, slots=True)
class AnswerDraft:
    """What the model produced, before any marker has been checked.

    "Draft" is the important word. The text in here still contains whatever
    markers the model chose to write, including ones that name no passage, and
    it must not be shown to anyone until it has been through
    ``app.services.answers``.
    """

    text: str
    answered: bool
    model: str
    input_tokens: int | None
    output_tokens: int | None


class AnswerModel(Protocol):
    """What the answer path needs from a language model, and nothing more."""

    @property
    def model_name(self) -> str: ...

    @property
    def config_error(self) -> str | None:
        """Why this model cannot be called at all, for readiness to report."""

    async def generate(self, question: str, passages: Sequence[str]) -> AnswerDraft:
        """Answer ``question`` from ``passages``, or raise a ``GenerationError``."""


class ClaudeAnswerModel:
    """Answers via the Anthropic Messages API."""

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        effort: Effort = DEFAULT_EFFORT,
        timeout: float = 60.0,
    ) -> None:
        # An empty string is treated as absent, because that is what
        # ANTHROPIC_API_KEY= in a .env produces and it plainly means "not
        # configured" rather than "configured with nothing". Without this, the
        # commented-out default in .env.example would buy a round trip to the
        # API and a 401 reported as a rejected key.
        self._api_key = (api_key or "").strip() or None
        self._model = model
        self._max_tokens = max_tokens
        self._effort = effort
        self._timeout = timeout
        self._client: anthropic.AsyncAnthropic | None = None

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def config_error(self) -> str | None:
        """Whether a key is present - not whether it works.

        Deliberately not a live call. A readiness probe that asked the model
        anything would cost money on every poll and add seconds to a check an
        orchestrator runs every few seconds. A rejected key is caught the first
        time somebody asks a question, and reported as such.
        """
        return None if self._api_key else NOT_CONFIGURED_DETAIL

    async def generate(self, question: str, passages: Sequence[str]) -> AnswerDraft:
        client = self._ensure_client()
        try:
            message = await client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": build_prompt(question, passages)}],
                # Thinking is left at its default (adaptive, which is on for
                # this model family) and depth is controlled with effort
                # instead. Disabling it is documented to make the model
                # occasionally leak internal tags into the visible text, and
                # that text is what the marker resolver parses.
                output_config={
                    "effort": self._effort,
                    "format": {"type": "json_schema", "schema": DRAFT_SCHEMA},
                },
            )
        # Caught at the top of the SDK's hierarchy and sorted out by name,
        # rather than as a chain of `except` clauses, so that the distinctions
        # - which of these is worth retrying, which means this deployment is
        # broken - can be tested without a network. Nothing else is caught: a
        # bug in this module should not be reported as an unavailable model.
        except anthropic.APIError as exc:
            raise describe_failure(exc) from exc

        return read_draft(message)

    async def close(self) -> None:
        """Release the HTTP connection pool. Called from the app lifespan."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    def _ensure_client(self) -> anthropic.AsyncAnthropic:
        """The SDK client, built on first use.

        Lazily, so that a deployment with no key still starts, still serves
        ``/health``, and reports the problem through readiness - the same
        bargain the embedding model gets - rather than crash-looping.
        """
        if self._api_key is None:
            raise GenerationError(NOT_CONFIGURED_DETAIL)

        if self._client is None:
            self._client = anthropic.AsyncAnthropic(
                api_key=self._api_key,
                timeout=self._timeout,
                max_retries=MAX_RETRIES,
            )
        return self._client


def describe_failure(exc: anthropic.APIError) -> GenerationError:
    """Say what an SDK exception means for the person who asked the question.

    The order of the checks is the point, and it is the same order a
    most-specific-first ``except`` chain would need: a rejected key, a rate
    limit and a bad request are all ``APIStatusError`` subclasses, and one
    broad branch would report a misconfigured deployment as a busy service and
    invite the user to retry it forever.
    """
    if isinstance(exc, anthropic.AuthenticationError | anthropic.PermissionDeniedError):
        # Not transient and not the asker's problem, but somebody has to see
        # it: this is a deployment that cannot answer anything until its key is
        # fixed.
        logger.error("Anthropic rejected our credentials: %s", exc)
        return GenerationError(REJECTED_DETAIL)
    if isinstance(exc, anthropic.RateLimitError):
        logger.warning("rate limited by Anthropic: %s", exc)
        return GenerationError(BUSY_DETAIL)
    if isinstance(exc, anthropic.APIStatusError):
        # 5xx is theirs and passes; 4xx is a request this module built wrong,
        # which the asker can do nothing about and we must not hide.
        if exc.status_code >= 500:
            logger.warning("Anthropic returned %d: %s", exc.status_code, exc)
            return GenerationError(BUSY_DETAIL)
        logger.error("Anthropic rejected the request (%d): %s", exc.status_code, exc)
        return GenerationError(FAILED_DETAIL)
    # APITimeoutError is a subclass of this one, so both land here, and for
    # both the answer is the same: it may well work in a moment.
    if isinstance(exc, anthropic.APIConnectionError):
        logger.warning("could not reach Anthropic: %s", exc)
        return GenerationError(BUSY_DETAIL)
    logger.error("unexpected Anthropic error: %r", exc)
    return GenerationError(FAILED_DETAIL)


def read_draft(message: anthropic.types.Message) -> AnswerDraft:
    """Turn a response into a draft, or explain why there is not one.

    The trust boundary, and the reason it is a function rather than a method:
    everything that can arrive here badly - a refusal, a truncation, output
    that does not match the schema - can be constructed in a test and asserted
    on without an API key.
    """
    # Checked before the content is touched. A refusal still arrives as a 200
    # with content blocks, and reading those as an answer would publish the
    # refusal as though the course materials had said it.
    if message.stop_reason == "refusal":
        category = message.stop_details.category if message.stop_details else None
        logger.warning("model refused to answer (category=%s)", category)
        raise UnusableAnswerError(REFUSED_DETAIL)
    if message.stop_reason == "max_tokens":
        # The JSON is cut off mid-string, so there is nothing to salvage.
        logger.warning("answer hit the token ceiling after %d tokens", message.usage.output_tokens)
        raise UnusableAnswerError(TRUNCATED_DETAIL)

    # Joined rather than "the first text block", because thinking blocks share
    # the list and a structured payload is not promised to arrive in one piece.
    payload = "".join(block.text for block in message.content if block.type == "text")
    try:
        draft = DraftAnswer.model_validate_json(payload)
    except ValidationError as exc:
        # output_config.format is supposed to make this impossible, so if it
        # happens the schema and the API have parted ways, and the whole
        # payload is worth having in the log.
        logger.error("model output did not match the schema: %s\n%s", exc, payload[:1000])
        raise UnusableAnswerError(MALFORMED_DETAIL) from exc

    return AnswerDraft(
        text=draft.answer,
        answered=draft.answered,
        # The model that actually served the request, read off the response
        # rather than assumed from the request: an alias resolves to a concrete
        # version, and the log should record which one answered.
        model=message.model,
        input_tokens=message.usage.input_tokens,
        output_tokens=message.usage.output_tokens,
    )


def build_prompt(question: str, passages: Sequence[str]) -> str:
    """Lay the passages out under their markers, then ask the question.

    Public because it is the thing worth asserting on: a test can check that
    no filename, page number or chunk id is anywhere in the string the model is
    sent, which is a stronger statement about the citation guarantee than any
    assertion about the answer.
    """
    block = "\n\n".join(
        f"[{marker(index)}]\n{text.strip()}" for index, text in enumerate(passages, start=1)
    )
    labels = ", ".join(f"[{marker(index)}]" for index in range(1, len(passages) + 1))
    return (
        f"<passages>\n{block}\n</passages>\n\n"
        f"The markers that exist are: {labels}. There are no others.\n\n"
        f"Question: {question}"
    )


def get_answer_model(request: Request) -> AnswerModel:
    """The app's answer model, as a FastAPI dependency.

    The override point for tests, and the reason nothing in the default suite
    spends money: replacing this replaces the model everywhere the answer path
    can reach it.
    """
    model: AnswerModel = request.app.state.answer_model
    return model
