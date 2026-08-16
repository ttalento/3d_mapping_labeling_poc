"""VLM response handling.

The truncation path is tested because it was originally silent: a reply cut at
the token limit surfaced as `JSONDecodeError: Unterminated string`, got retried
four times, and burned free-tier quota on every attempt.
"""

import json
from types import SimpleNamespace

import pytest

from room3d.vlm import (
    RESPONSE_SCHEMA,
    Detection,
    GeminiDetector,
    VLMError,
    _raise_if_truncated,
    _schema_requiring_mask,
    _Truncated,
)


def response(finish_reason):
    return SimpleNamespace(candidates=[SimpleNamespace(finish_reason=finish_reason)])


# --- truncation --------------------------------------------------------------


def test_max_tokens_finish_reason_raises():
    with pytest.raises(_Truncated, match="truncated"):
        _raise_if_truncated(response("MAX_TOKENS"), request_masks=False)


def test_truncation_blames_masks_when_masks_were_requested():
    with pytest.raises(_Truncated, match="use_masks"):
        _raise_if_truncated(response("MAX_TOKENS"), request_masks=True)


def test_truncation_suggests_token_budget_when_masks_were_not_requested():
    with pytest.raises(_Truncated, match="max_output_tokens"):
        _raise_if_truncated(response("MAX_TOKENS"), request_masks=False)


def test_normal_finish_reason_passes():
    _raise_if_truncated(response("STOP"), request_masks=False)


def test_missing_candidates_is_not_an_error():
    _raise_if_truncated(SimpleNamespace(candidates=[]), request_masks=False)
    _raise_if_truncated(SimpleNamespace(), request_masks=False)


def test_truncated_is_a_vlm_error():
    """So callers that catch VLMError still handle it."""
    assert issubclass(_Truncated, VLMError)


# --- schema ------------------------------------------------------------------


def test_default_schema_does_not_require_mask():
    assert "mask" not in RESPONSE_SCHEMA["properties"]["objects"]["items"]["required"]


def test_mask_schema_requires_mask():
    """An optional mask is simply omitted by the model, so it must be required."""
    schema = _schema_requiring_mask()
    assert "mask" in schema["properties"]["objects"]["items"]["required"]


def test_mask_schema_does_not_mutate_the_shared_default():
    _schema_requiring_mask()
    assert "mask" not in RESPONSE_SCHEMA["properties"]["objects"]["items"]["required"]


# --- parsing -----------------------------------------------------------------


def parse(payload):
    return GeminiDetector._parse(json.dumps(payload))


def test_parse_reads_a_well_formed_response():
    dets = parse({"objects": [
        {"label": "desk", "box_2d": [10, 20, 30, 40], "confidence": 0.9}
    ]})
    assert len(dets) == 1
    assert dets[0] == Detection(label="desk", box_2d=(10, 20, 30, 40), confidence=0.9)


def test_parse_skips_malformed_entries_without_failing_the_frame():
    """One bad object should not cost the other twelve."""
    dets = parse({"objects": [
        {"label": "desk", "box_2d": [1, 2, 3, 4], "confidence": 0.9},
        {"label": "broken", "box_2d": [1, 2, 3]},          # too few coords
        {"box_2d": [1, 2, 3, 4], "confidence": 0.5},        # no label
        {"label": "", "box_2d": [1, 2, 3, 4]},              # empty label
        {"label": "nonnumeric", "box_2d": ["a", "b", "c", "d"]},
        {"label": "lamp", "box_2d": [5, 6, 7, 8], "confidence": 0.7},
    ]})
    assert [d.label for d in dets] == ["desk", "lamp"]


def test_parse_defaults_and_clamps_confidence():
    dets = parse({"objects": [
        {"label": "a", "box_2d": [1, 2, 3, 4]},
        {"label": "b", "box_2d": [1, 2, 3, 4], "confidence": 5.0},
        {"label": "c", "box_2d": [1, 2, 3, 4], "confidence": -1.0},
    ]})
    assert [d.confidence for d in dets] == [0.5, 1.0, 0.0]


def test_parse_rounds_float_coordinates():
    dets = parse({"objects": [
        {"label": "a", "box_2d": [1.6, 2.4, 3.5, 4.2], "confidence": 0.5}
    ]})
    assert dets[0].box_2d == (2, 2, 4, 4)


def test_parse_strips_label_whitespace():
    dets = parse({"objects": [
        {"label": "  office chair  ", "box_2d": [1, 2, 3, 4], "confidence": 0.5}
    ]})
    assert dets[0].label == "office chair"


def test_parse_treats_empty_mask_as_absent():
    dets = parse({"objects": [
        {"label": "a", "box_2d": [1, 2, 3, 4], "confidence": 0.5, "mask": ""}
    ]})
    assert dets[0].mask_b64 is None


def test_parse_handles_no_objects_key():
    assert GeminiDetector._parse("{}") == []


def test_parse_rejects_non_json():
    with pytest.raises(VLMError, match="not valid JSON"):
        GeminiDetector._parse("Unterminated string starting at")
