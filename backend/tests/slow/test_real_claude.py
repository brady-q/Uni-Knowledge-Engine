"""The real Claude API, called twice. Excluded from the default suite.

Everything else about answering is tested against ``tests.answers``, which is
what keeps `make test` free, offline and deterministic. That trade has exactly
one hole in it, and this file is the patch: a fake cannot tell us whether the
*prompt* works. Specifically, whether a real model asked to answer only from
six labelled passages will

* cite the marker of the passage it actually used,
* invent no marker that was not supplied, and
* say it does not know, rather than answer from what it knows anyway,

and whether ``output_config.format`` really does return JSON matching
``DraftAnswer``. If any of that has drifted - a prompt edit, a model change, an
API change - the whole suite stays green and the product quietly starts
producing uncited or wrongly cited answers.

The passages are invented on purpose. A real model knows what gradient descent
is, so a question about that would be answered correctly whether it read the
passages or not, and the test would prove nothing. The facts below exist
nowhere but in this file, so an answer containing them can only have come from
the passages it was given.

Run with:

    make test-live      # or: pytest -m live

It costs a few cents and needs ANTHROPIC_API_KEY; it skips without one.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import pytest

from app.config import DEFAULT_ANSWER_MODEL, Settings
from app.llm import ClaudeAnswerModel
from app.repositories.chunks import ChunkMatch
from app.services.answers import resolve_markers
from tests.answers import make_match

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY is not set"),
]

# Facts that exist nowhere else, so an answer carrying them was read rather
# than recalled. The order is the ranking the retrieval would have produced.
PASSAGES = [
    "The Frobisher coursework in this unit is marked out of 40 and submitted "
    "through the departmental portal.",
    "Attendance at the Frobisher laboratory sessions is not assessed, though "
    "the demonstrators keep a register for their own planning.",
    "The Frobisher coursework deadline is the Thursday of teaching week nine, "
    "at four in the afternoon.",
    "Extensions to the Frobisher coursework are granted only by the unit "
    "director, and never retrospectively.",
]


@pytest.fixture
async def model() -> AsyncIterator[ClaudeAnswerModel]:
    """The real client, configured exactly as the app configures it.

    Built from ``Settings`` rather than from literals so that this test breaks
    if the deployed defaults drift away from what has actually been tried
    against the API - which is the only reason it is worth its cost.
    """
    settings = Settings(database_url="postgresql+asyncpg://unused/unused")
    answerer = ClaudeAnswerModel(
        api_key=settings.anthropic_api_key,
        model=settings.answer_model,
        max_tokens=settings.answer_max_tokens,
        effort=settings.answer_effort,
        timeout=settings.answer_timeout_seconds,
    )
    try:
        yield answerer
    finally:
        await answerer.close()


def matches() -> list[ChunkMatch]:
    """The passages as retrieved rows, so resolution can be run for real."""
    return [make_match(text=text, page_start=index + 1) for index, text in enumerate(PASSAGES)]


def test_the_configured_model_is_the_one_this_test_was_written_against() -> None:
    """A model swap should have to come past this file. What the prompt asks
    for is not guaranteed to survive a change of model, and a swap made in the
    environment alone would be invisible to every other test."""
    assert Settings(database_url="postgresql+asyncpg://unused/unused").answer_model == (
        DEFAULT_ANSWER_MODEL
    )


async def test_it_answers_from_the_passages_and_cites_the_right_one(
    model: ClaudeAnswerModel,
) -> None:
    """One real call, and the four things that have to hold about it."""
    draft = await model.generate("When is the Frobisher coursework due?", PASSAGES)

    answer, citations, dropped = resolve_markers(draft.text, matches())

    # It read the passages: "week nine" appears only in the third of them.
    assert "nine" in answer.lower()
    assert draft.answered is True
    # It cited, and it cited the passage the fact is actually in.
    assert [citation.marker for citation in citations] == [3]
    # It invented nothing. This is the assertion the design exists to make
    # possible, and the only one that needs a real model to be worth anything.
    assert dropped == 0
    # It wrote no page number and no filename, because it was shown neither.
    for invented in ("page", ".pdf", "handbook", "lecture"):
        assert invented not in answer.lower(), answer
    assert draft.model.startswith("claude-")
    assert draft.input_tokens and draft.output_tokens


async def test_it_declines_rather_than_answering_from_general_knowledge(
    model: ClaudeAnswerModel,
) -> None:
    """The other half of the prompt's job, and the harder half.

    The question is one any model could answer confidently from what it knows.
    Nothing in the passages touches it, so the only correct behaviour is to say
    so - with ``answered`` false and no markers, since there is nothing to
    point at.
    """
    draft = await model.generate("What is the capital city of France?", PASSAGES)

    answer, citations, dropped = resolve_markers(draft.text, matches())

    assert draft.answered is False
    assert citations == []
    assert dropped == 0
    assert "paris" not in answer.lower()
