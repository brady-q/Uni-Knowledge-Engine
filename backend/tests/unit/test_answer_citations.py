"""Marker resolution and prompt construction: the citation guarantee itself.

These are the two pure functions the whole design rests on, and neither needs a
database, a model or an app to test - which is the point. Everything else in
the answer path is plumbing around them:

* ``build_prompt`` decides what the model is allowed to see. If a page number
  can reach it, the model can repeat one, and no amount of downstream
  resolution can tell an invented page number from a remembered one.
* ``resolve_markers`` decides what survives. A marker naming a retrieved
  passage becomes a citation built from the stored rows; a marker naming
  nothing is deleted from the prose.

The failure this file is written to catch is the quiet one: an answer that
still reads confidently after its evidence has been removed.
"""

from __future__ import annotations

from app.llm import build_prompt
from app.services.answers import resolve_markers
from tests.answers import make_match

# --- what the model is shown --------------------------------------------------


def test_the_prompt_labels_every_passage_in_rank_order() -> None:
    prompt = build_prompt("why?", ["first passage", "second passage", "third passage"])

    assert prompt.index("[S1]") < prompt.index("[S2]") < prompt.index("[S3]")
    assert prompt.index("first passage") < prompt.index("second passage")


def test_the_prompt_contains_nothing_a_citation_is_made_of() -> None:
    """The load-bearing test in this file.

    Filenames, page numbers, chunk ids and document ids are all read out of the
    database when a citation is rendered. None of them is in the model's
    context, so none of them is something it could get wrong - which is a
    property of the interface rather than of the model's behaviour, and this is
    where that property is checked.
    """
    match = make_match(text="Gradient descent takes a step downhill.", page_start=11, page_end=12)

    prompt = build_prompt("what is gradient descent?", [match.chunk.text])

    for forbidden in (
        match.document.filename,
        str(match.chunk.id),
        str(match.document.id),
        "page",
        "11",
    ):
        assert forbidden not in prompt, forbidden


def test_the_prompt_names_the_markers_that_exist() -> None:
    """Stated as well as implied, because "do not invent one" is only
    actionable next to the list of the ones that are real."""
    prompt = build_prompt("why?", ["a", "b"])

    assert "The markers that exist are: [S1], [S2]." in prompt


def test_the_question_is_last() -> None:
    """Passages first, question last: the instruction the model acts on should
    not be buried above several hundred words of course material."""
    prompt = build_prompt("when is it due?", ["some material"])

    assert prompt.rstrip().endswith("Question: when is it due?")


# --- resolution ---------------------------------------------------------------


def test_a_valid_marker_resolves_to_the_row_it_named() -> None:
    matches = [
        make_match(text="first", filename="week1.pdf", page_start=2, page_end=2),
        make_match(text="second", filename="week2.pdf", page_start=7, page_end=8),
    ]

    answer, citations, dropped = resolve_markers("It is in the second reading [S2].", matches)

    assert answer == "It is in the second reading [S2]."
    assert dropped == 0
    assert [citation.marker for citation in citations] == [2]
    assert citations[0].document.filename == "week2.pdf"
    assert (citations[0].chunk.page_start, citations[0].chunk.page_end) == (7, 8)


def test_citations_come_back_in_marker_order_not_prose_order() -> None:
    """Marker order is retrieval order, so a citation list is ranked by
    relevance whatever order the model happened to mention things in."""
    matches = [make_match(text=str(index)) for index in range(3)]

    _, citations, _ = resolve_markers("Third [S3], then first [S1], then second [S2].", matches)

    assert [citation.marker for citation in citations] == [1, 2, 3]


def test_a_marker_used_twice_is_one_citation() -> None:
    """The same source, mentioned twice. A footnote list should not repeat it -
    but both mentions stay in the prose, where the claims are."""
    matches = [make_match(text="only passage")]

    answer, citations, dropped = resolve_markers("Yes [S1], and also later [S1].", matches)

    assert answer == "Yes [S1], and also later [S1]."
    assert len(citations) == 1
    assert dropped == 0


def test_an_invented_marker_is_deleted_from_the_answer() -> None:
    """The model was shown two passages and cited a seventh."""
    matches = [make_match(text="first"), make_match(text="second")]

    answer, citations, dropped = resolve_markers(
        "It is due in week six [S7], per the schedule [S1].", matches
    )

    assert answer == "It is due in week six, per the schedule [S1]."
    assert dropped == 1
    assert [citation.marker for citation in citations] == [1]


def test_deleting_a_marker_does_not_leave_a_space_before_the_full_stop() -> None:
    """Cosmetic, but the answer is rendered as prose to a reader, and "shown ."
    reads as a bug in the product rather than as a dropped citation."""
    matches = [make_match(text="first")]

    answer, _, _ = resolve_markers("As shown [S4]. And again [S9]!", matches)

    assert answer == "As shown. And again!"


def test_every_invented_occurrence_is_counted() -> None:
    """Occurrences, not distinct markers: the number answers "how often did the
    model make a citation up?", which is the question worth logging."""
    matches = [make_match(text="first")]

    _, citations, dropped = resolve_markers("One [S8], two [S8], three [S9].", matches)

    assert dropped == 3
    assert citations == []


def test_a_zero_marker_is_invented() -> None:
    """Passages are 1-based, so [S0] names nothing however plausible it looks."""
    matches = [make_match(text="first")]

    answer, citations, dropped = resolve_markers("Nowhere [S0].", matches)

    assert answer == "Nowhere."
    assert (citations, dropped) == ([], 1)


def test_an_absurd_marker_does_not_raise() -> None:
    """The pattern accepts any run of digits, so resolution has to survive one
    that would not fit in an index."""
    matches = [make_match(text="first")]

    answer, citations, dropped = resolve_markers("Somewhere [S99999999999999999999].", matches)

    assert answer == "Somewhere."
    assert (citations, dropped) == ([], 1)


def test_text_without_markers_resolves_to_nothing() -> None:
    """What an honest abstention looks like: prose, no markers, no citations.
    The service is what decides whether that is an answer or a refusal."""
    matches = [make_match(text="first")]

    answer, citations, dropped = resolve_markers("The course materials do not cover that.", matches)

    assert answer == "The course materials do not cover that."
    assert (citations, dropped) == ([], 0)


def test_markers_are_resolved_even_with_no_passages_to_resolve_against() -> None:
    """Cannot arise today - an empty retrieval short-circuits before the model
    is called - but the function must not index into an empty list if it does."""
    answer, citations, dropped = resolve_markers("Confidently [S1].", [])

    assert answer == "Confidently."
    assert (citations, dropped) == ([], 1)
