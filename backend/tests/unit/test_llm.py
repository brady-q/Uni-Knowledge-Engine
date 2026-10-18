"""Reading a response, and explaining a failure, with no API key in sight.

``app.llm`` is the module that talks to Anthropic, and almost none of it needs
a network to test. The two functions here are the ones that decide what the
rest of the system believes:

* ``read_draft`` is the trust boundary. A refusal, an answer cut off at the
  token ceiling and output that does not match the schema all arrive as an
  ordinary ``200``, and every one of them has to become an error rather than
  an answer. Publishing a refusal as though the course material had said it is
  the worst outcome available here.
* ``describe_failure`` decides what the asker is told. The distinctions matter
  in a way a status code cannot express - a rejected key is not a busy service,
  and telling somebody to retry a misconfigured deployment wastes their time
  forever - which is precisely why it is a named function and not an
  ``except`` chain.

``httpx2`` rather than ``httpx`` below: the Anthropic SDK is built on it, and
its exceptions carry a request and a response of that type.
"""

from __future__ import annotations

import anthropic
import httpx2
import pytest
from anthropic.types import (
    Message,
    RefusalStopDetails,
    StopReason,
    TextBlock,
    ThinkingBlock,
    Usage,
)

from app.errors import GenerationError, UnusableAnswerError
from app.llm import (
    BUSY_DETAIL,
    DRAFT_SCHEMA,
    FAILED_DETAIL,
    MALFORMED_DETAIL,
    NOT_CONFIGURED_DETAIL,
    REFUSED_DETAIL,
    REJECTED_DETAIL,
    TRUNCATED_DETAIL,
    ClaudeAnswerModel,
    describe_failure,
    read_draft,
)

REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def make_message(
    *,
    text: str = '{"answer": "Week six [S1].", "answered": true}',
    stop_reason: StopReason = "end_turn",
    model: str = "claude-opus-5-20260101",
    input_tokens: int = 1_234,
    output_tokens: int = 56,
    blocks: list[TextBlock | ThinkingBlock] | None = None,
    stop_details: RefusalStopDetails | None = None,
) -> Message:
    return Message(
        id="msg_01",
        type="message",
        role="assistant",
        model=model,
        stop_reason=stop_reason,
        stop_details=stop_details,
        content=blocks if blocks is not None else [TextBlock(type="text", text=text)],
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def status_error(
    code: int, kind: type[anthropic.APIStatusError] = anthropic.APIStatusError
) -> anthropic.APIStatusError:
    return kind(
        message=f"upstream said {code}",
        response=httpx2.Response(code, request=REQUEST),
        body=None,
    )


# --- the schema sent with every request ---------------------------------------


def test_the_schema_forbids_fields_the_prompt_did_not_ask_for() -> None:
    """Structured outputs requires it, and it is what makes the Pydantic model
    and the wire format the same contract rather than two similar ones."""
    assert DRAFT_SCHEMA["additionalProperties"] is False
    assert sorted(DRAFT_SCHEMA["required"]) == ["answer", "answered"]


def test_the_schema_carries_no_prose_written_for_maintainers() -> None:
    """Pydantic promotes the class docstring into the schema, and the schema is
    part of the prompt. The rationale in that docstring is addressed to whoever
    edits the file; the model should not be reading it as an instruction."""
    assert "description" not in DRAFT_SCHEMA
    assert "title" not in DRAFT_SCHEMA
    # Field descriptions do survive: those are written for the model.
    assert "[S#]" in DRAFT_SCHEMA["properties"]["answer"]["description"]


# --- reading a response -------------------------------------------------------


def test_a_well_formed_response_becomes_a_draft() -> None:
    draft = read_draft(make_message())

    assert draft.text == "Week six [S1]."
    assert draft.answered is True
    assert (draft.input_tokens, draft.output_tokens) == (1_234, 56)


def test_the_model_recorded_is_the_one_that_answered() -> None:
    """Read off the response, not assumed from the request: an alias resolves
    to a concrete version, and the query log should say which one spoke."""
    draft = read_draft(make_message(model="claude-opus-5-20260101"))

    assert draft.model == "claude-opus-5-20260101"


def test_an_abstention_is_a_draft_like_any_other() -> None:
    """Not an error. "The materials do not cover this" is a real answer, and
    the service logs it the same way it logs one with citations."""
    draft = read_draft(
        make_message(
            text='{"answer": "The course materials do not cover that.", "answered": false}'
        )
    )

    assert draft.answered is False
    assert "[S" not in draft.text


def test_thinking_blocks_are_not_part_of_the_payload() -> None:
    """They share the content list with the answer, and concatenating one into
    the JSON would fail validation for a response that was perfectly fine."""
    message = make_message(
        blocks=[
            ThinkingBlock(type="thinking", thinking="", signature="sig"),
            TextBlock(type="text", text='{"answer": "Yes [S2].", "answered": true}'),
        ]
    )

    assert read_draft(message).text == "Yes [S2]."


def test_a_refusal_is_never_served_as_an_answer() -> None:
    """A refusal is a 200 with content in it. Returning that content would put
    the model's own words in the mouth of the course material."""
    message = make_message(
        stop_reason="refusal",
        stop_details=RefusalStopDetails(type="refusal", category="cyber"),
        text="I can't help with that.",
    )

    with pytest.raises(UnusableAnswerError) as caught:
        read_draft(message)

    assert caught.value.detail == REFUSED_DETAIL


def test_a_refusal_without_details_is_still_a_refusal() -> None:
    """``stop_details`` is only populated for refusals and may still be null,
    so reading the category has to be guarded."""
    with pytest.raises(UnusableAnswerError):
        read_draft(make_message(stop_reason="refusal", stop_details=None))


def test_an_answer_cut_off_at_the_ceiling_is_an_error_not_a_fragment() -> None:
    """The JSON stops mid-string, so there is no half-answer to salvage - and
    the advice that helps is "ask something narrower", not "try again"."""
    with pytest.raises(UnusableAnswerError) as caught:
        read_draft(make_message(stop_reason="max_tokens", text='{"answer": "Week si'))

    assert caught.value.detail == TRUNCATED_DETAIL


def test_output_that_misses_the_schema_is_rejected() -> None:
    """output_config.format is supposed to make this impossible. If it ever
    happens, the schema and the API have parted ways, and guessing at the
    payload is how an uncited answer gets served."""
    with pytest.raises(UnusableAnswerError) as caught:
        read_draft(make_message(text='{"answer": "no verdict field"}'))

    assert caught.value.detail == MALFORMED_DETAIL


def test_prose_instead_of_json_is_rejected() -> None:
    with pytest.raises(UnusableAnswerError):
        read_draft(make_message(text="Sure! Here is your answer: week six."))


def test_an_empty_response_is_rejected() -> None:
    with pytest.raises(UnusableAnswerError):
        read_draft(make_message(blocks=[]))


# --- explaining a failure -----------------------------------------------------


def test_a_rejected_key_is_not_reported_as_a_busy_service() -> None:
    """The distinction this function exists for: one of these fixes itself and
    the other needs somebody to edit an environment variable."""
    rejected = describe_failure(status_error(401, anthropic.AuthenticationError))

    assert rejected.detail == REJECTED_DETAIL
    assert "try again" not in rejected.detail


def test_a_forbidden_key_reads_the_same_as_a_rejected_one() -> None:
    """A key that exists but may not use this model is still a configuration
    problem, and the asker cannot tell the two apart or act on either."""
    assert describe_failure(status_error(403, anthropic.PermissionDeniedError)).detail == (
        REJECTED_DETAIL
    )


def test_a_rate_limit_is_worth_retrying() -> None:
    assert describe_failure(status_error(429, anthropic.RateLimitError)).detail == BUSY_DETAIL


def test_an_upstream_server_error_is_worth_retrying() -> None:
    for code in (500, 502, 529):
        assert describe_failure(status_error(code)).detail == BUSY_DETAIL, code


def test_a_bad_request_is_our_bug_and_says_so() -> None:
    """Nothing the asker can do about it, and nothing that retrying will fix,
    so the detail points at the logs rather than at them."""
    detail = describe_failure(status_error(422, anthropic.UnprocessableEntityError)).detail

    assert detail == FAILED_DETAIL
    assert "try again" not in detail


def test_a_timeout_is_worth_retrying() -> None:
    assert describe_failure(anthropic.APITimeoutError(request=REQUEST)).detail == BUSY_DETAIL


def test_an_unreachable_api_is_worth_retrying() -> None:
    assert describe_failure(anthropic.APIConnectionError(request=REQUEST)).detail == BUSY_DETAIL


def test_every_failure_is_a_generation_error() -> None:
    """So that the answer service can name one type, and app.main can map it
    to one status, however the SDK's hierarchy grows."""
    for exc in (
        status_error(401, anthropic.AuthenticationError),
        status_error(429, anthropic.RateLimitError),
        status_error(500),
        anthropic.APIConnectionError(request=REQUEST),
    ):
        assert isinstance(describe_failure(exc), GenerationError)


# --- configuration ------------------------------------------------------------


async def test_no_key_means_no_call_is_attempted() -> None:
    """Not an authentication failure at the API: this never leaves the process,
    which is what makes it safe for the test suite to run at all."""
    model = ClaudeAnswerModel(api_key=None, model="claude-opus-5")

    with pytest.raises(GenerationError) as caught:
        await model.generate("why?", ["some passage"])

    assert caught.value.detail == NOT_CONFIGURED_DETAIL


def test_a_missing_key_is_reported_to_readiness() -> None:
    model = ClaudeAnswerModel(api_key=None, model="claude-opus-5")

    assert model.config_error == NOT_CONFIGURED_DETAIL
    assert model.model_name == "claude-opus-5"


def test_an_empty_key_is_treated_as_no_key() -> None:
    """``ANTHROPIC_API_KEY=`` in a .env sets the variable to the empty string,
    which means "not configured" - not "configured with nothing", which would
    buy a round trip to the API and a 401 reported as a rejected key."""
    assert ClaudeAnswerModel(api_key="", model="m").config_error == NOT_CONFIGURED_DETAIL
    assert ClaudeAnswerModel(api_key="   ", model="m").config_error == NOT_CONFIGURED_DETAIL


def test_a_present_key_is_all_readiness_can_check() -> None:
    """It cannot tell a valid key from an invalid one without spending money on
    every poll, and it deliberately does not try."""
    model = ClaudeAnswerModel(api_key="sk-ant-not-a-real-key", model="claude-opus-5")

    assert model.config_error is None


async def test_closing_a_model_that_was_never_used_is_harmless() -> None:
    """The lifespan closes it unconditionally, and most deployments of this
    will never have built a client at all."""
    await ClaudeAnswerModel(api_key=None, model="claude-opus-5").close()
