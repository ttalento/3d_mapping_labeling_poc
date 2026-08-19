"""Gemini object detection: labels with 2D boxes, and masks where obtainable.

A note on masks, because the plan called for them and measurement changed the
answer. A box round a chair contains a lot of the wall behind it, so masks
should give better 3D centroids. In practice, against gemini-3.5-flash-lite and
gemini-3.6-flash:

  * with `mask` optional in the response schema, the model simply omits it;
  * with `mask` required and base64 PNG requested, the response exceeds the
    output token budget and truncates mid-string, producing invalid JSON;
  * with no response schema at all, masks come back as COCO RLE (not PNG) and
    the reply still runs to ~65k characters and truncates.

So masks are off by default. Boxes plus the depth clustering in projection.py
are the reliable path: clustering is precisely the mechanism that stops a
box-shaped selection from dragging the centroid onto the background wall, which
is the job the mask would otherwise have done. Mask support is kept behind
`use_masks` because a future model will make it work, and the projection code
already accepts either.

The call is made with google-genai directly rather than through CrewAI's
multimodal agent path, which is awkward to pin to a strict schema. The agent
decides *which* frames to spend calls on; this module is the instrument it uses.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Object detection with segmentation masks is a spatial-grounding task, which
# the non-lite flash models do noticeably better. MODEL/GEMINI_MODEL in .env
# overrides this.
DEFAULT_MODEL = "gemini-3.6-flash"

DETECTION_PROMPT = """\
You are surveying a single room to build a 3D map of its contents.

Detect the prominent, physically distinct objects in this image: furniture,
appliances, fixtures, and sizeable items sitting on surfaces.

Rules:
- One entry per physical object. If there are three chairs, return three entries.
- Use a short, common noun for `label` ("desk", "monitor", "office chair").
- Do NOT return architectural surfaces: wall, floor, ceiling. DO return windows and doors.
- Skip anything smaller than roughly 10 cm, and anything you are guessing at.
- `box_2d` must be [ymin, xmin, ymax, xmax], normalised 0-1000.
- `confidence` is your own 0-1 certainty that the label is correct.

Return at most 25 objects."""

LOCATE_PROMPT = """\
You are looking at one frame of a walkthrough of a single room.

Find every physically distinct object in this image matching this description:

    {phrase}

Rules:
- One entry per physical object. If two things match the description, return two.
- If the description mentions a position or a relationship ("by the window", "the
  left one"), only return objects that actually satisfy it.
- If nothing in this image matches, return an empty list. An empty list is a
  correct and useful answer; a guess is not.
- Set `label` to the description you were given, verbatim.
- `box_2d` must be [ymin, xmin, ymax, xmax], normalised 0-1000.
- `confidence` is your own 0-1 certainty that this object matches the description.

Return at most 5 objects."""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "box_2d": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "mask": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["label", "box_2d", "confidence"],
            },
        }
    },
    "required": ["objects"],
}


@dataclass
class Detection:
    label: str
    box_2d: tuple[int, int, int, int]     # ymin, xmin, ymax, xmax in 0-1000
    confidence: float
    mask_b64: str | None = None


class VLMError(RuntimeError):
    pass


class _Truncated(VLMError):
    """The reply hit the output token cap, so the JSON is cut mid-string."""


def _schema_requiring_mask() -> dict:
    schema = json.loads(json.dumps(RESPONSE_SCHEMA))     # deep copy
    schema["properties"]["objects"]["items"]["required"] = [
        "label", "box_2d", "mask", "confidence"
    ]
    return schema


def _raise_if_truncated(response, request_masks: bool) -> None:
    """Turn a token-limit truncation into a message that says what to do.

    Without this, a truncated reply surfaces as `JSONDecodeError: Unterminated
    string`, is retried four times, and burns quota on every attempt.
    """
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return

    reason = str(getattr(candidates[0], "finish_reason", "") or "")
    if "MAX_TOKENS" not in reason.upper():
        return

    hint = (
        " Segmentation masks are the usual cause: they do not fit in the output "
        "budget. Set label.use_masks: false in your config."
        if request_masks
        else " Try raising max_output_tokens or lowering the object count in the prompt."
    )
    raise _Truncated(f"response truncated at the output token limit ({reason}).{hint}")


class GeminiDetector:
    """Thin, retrying wrapper around Gemini's image understanding."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        *,
        request_masks: bool = False,
        max_retries: int = 4,
        min_interval_s: float = 4.0,
        max_output_tokens: int = 8192,
    ):
        from google import genai

        key = api_key or resolve_api_key()
        if not key:
            raise VLMError(
                "No Gemini API key. Put GEMINI_API_KEY=... (or API_KEY=...) in .env "
                "at the repo root."
            )

        self._client = genai.Client(api_key=key)
        self.model = model
        self.request_masks = request_masks
        self.max_retries = max_retries
        self.min_interval_s = min_interval_s      # free tier is ~10-15 RPM
        self.max_output_tokens = max_output_tokens
        self._last_call = 0.0

    def detect(self, image: np.ndarray | Path | str) -> list[Detection]:
        """Detect objects in one image (RGB uint8 array, or a path)."""
        payload = self._to_part(image)
        prompt = DETECTION_PROMPT
        if self.request_masks:
            prompt += (
                "\n- Also return `mask`: the segmentation mask for the object as a "
                "base64-encoded PNG probability map covering exactly the box_2d region."
            )

        raw = self._call_with_retry(prompt, payload)
        return self._parse(raw)

    def locate(self, image: np.ndarray | Path | str, phrase: str) -> list[Detection]:
        """Find one named thing, rather than surveying everything.

        Two differences from `detect` that matter. The model is told what to look
        for, so it can resolve a description the cached labels cannot -- "the
        couch by the window" is a spatial relation, and labels do not carry
        those. And it is told that finding nothing is a correct answer, because
        the alternative is a model that always returns something.
        """
        payload = self._to_part(image)
        raw = self._call_with_retry(LOCATE_PROMPT.format(phrase=phrase), payload)
        return self._parse(raw)

    # --- internals ----------------------------------------------------------

    def _to_part(self, image):
        from google.genai import types

        if isinstance(image, (str, Path)):
            data = Path(image).read_bytes()
            suffix = Path(image).suffix.lower()
            mime = "image/png" if suffix == ".png" else "image/jpeg"
            return types.Part.from_bytes(data=data, mime_type=mime)

        import cv2

        arr = np.asarray(image)
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
        ok, buf = cv2.imencode(".png", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        if not ok:
            raise VLMError("failed to encode image")
        return types.Part.from_bytes(data=buf.tobytes(), mime_type="image/png")

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.min_interval_s:
            time.sleep(self.min_interval_s - elapsed)
        self._last_call = time.monotonic()

    def _call_with_retry(self, prompt: str, payload) -> str:
        from google.genai import types

        schema = dict(RESPONSE_SCHEMA)
        if self.request_masks:
            # The model omits an optional `mask`, so it has to be required to
            # appear at all -- at the cost of a much larger reply.
            schema = _schema_requiring_mask()

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.0,
            max_output_tokens=self.max_output_tokens,
        )

        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                self._throttle()
                response = self._client.models.generate_content(
                    model=self.model, contents=[payload, prompt], config=config
                )
                _raise_if_truncated(response, self.request_masks)
                if not response.text:
                    raise VLMError("empty response")
                return response.text
            except _Truncated:
                raise      # retrying a token-limit truncation just wastes quota
            except Exception as exc:  # noqa: BLE001 - retry on anything transient
                last = exc
                if attempt == self.max_retries - 1:
                    break
                # Free-tier 429s are the common case; back off generously.
                backoff = min(60.0, 2.0**attempt * 4.0) + random.uniform(0, 2)
                print(f"  [vlm] attempt {attempt + 1} failed ({exc}); retrying in {backoff:.0f}s")
                time.sleep(backoff)

        raise VLMError(f"Gemini call failed after {self.max_retries} attempts: {last}")

    @staticmethod
    def _parse(raw: str) -> list[Detection]:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise VLMError(f"response was not valid JSON: {exc}") from exc

        detections: list[Detection] = []
        for item in data.get("objects", []):
            box = item.get("box_2d")
            label = item.get("label")
            if not label or not box or len(box) != 4:
                continue          # skip malformed entries rather than fail the frame
            try:
                ymin, xmin, ymax, xmax = (int(round(float(v))) for v in box)
            except (TypeError, ValueError):
                continue

            detections.append(
                Detection(
                    label=str(label).strip(),
                    box_2d=(ymin, xmin, ymax, xmax),
                    confidence=float(np.clip(float(item.get("confidence", 0.5)), 0.0, 1.0)),
                    mask_b64=item.get("mask") or None,
                )
            )
        return detections


# Checked in order. GEMINI_API_KEY is the documented name; API_KEY is accepted
# because a .env for a single-provider project often just says API_KEY.
API_KEY_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "API_KEY")
MODEL_VARS = ("GEMINI_MODEL", "MODEL")


def resolve_api_key() -> str | None:
    for name in API_KEY_VARS:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


# A model name may arrive carrying a routing prefix: "models/" from the REST
# API, or "google/" and "gemini/" from LiteLLM-style routers. The google-genai
# SDK wants the bare name; CrewAI's LiteLLM wants "gemini/" back on the front.
# Normalising to bare here keeps one canonical form and one place to re-prefix.
_MODEL_PREFIXES = ("models/", "google/", "gemini/")


def strip_model_prefix(model: str) -> str:
    model = model.strip()
    changed = True
    while changed:
        changed = False
        for prefix in _MODEL_PREFIXES:
            if model.lower().startswith(prefix):
                model = model[len(prefix) :]
                changed = True
    return model


def resolve_model(default: str = DEFAULT_MODEL) -> str:
    """Bare model name from the environment, without any routing prefix."""
    for name in MODEL_VARS:
        value = os.environ.get(name)
        if value and value.strip():
            return strip_model_prefix(value)
    return default


def load_env(repo_root: Path | None = None) -> None:
    """Load .env from the repo root so the API key is picked up."""
    from dotenv import load_dotenv

    root = repo_root or Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env")
