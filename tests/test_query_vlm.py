"""Gate 8d: the VLM path, with the network stubbed out.

The rule under test is when Gemini is called at all. Calling it when the cache
could have answered wastes a scarce quota; not calling it when the cache cannot
answer produces a confident wrong "not found".
"""

import json

import numpy as np

from room3d.artifacts import save_frames_npz
from room3d.crew.pipeline import build_observations_doc
from room3d.crew.tools import ClusterObservationsTool, ProjectDetectionsTool
from room3d.query import query_room
from room3d.vlm import Detection
from test_boxfit_wiring import labelled_session, sofa_detection


class StubDetector:
    """Returns the sofa's box for any phrase, and counts every call."""

    def __init__(self, detection=None):
        self.calls = []
        self._detection = detection

    def locate(self, image, phrase):
        self.calls.append(phrase)
        det = self._detection if self._detection is not None else sofa_detection()
        return [Detection(label=phrase, box_2d=det.box_2d, confidence=0.85)]


def room_on_disk(tmp_path):
    session = labelled_session()
    ProjectDetectionsTool(session)._run()
    ClusterObservationsTool(session, None)._run(use_llm_synonyms=False)
    save_frames_npz(tmp_path / "frames.npz", session.recon)
    (tmp_path / "observations.json").write_text(
        json.dumps(build_observations_doc(session, "room"))
    )
    (tmp_path / "objects.json").write_text(json.dumps({"room": "room", "objects": []}))
    return tmp_path


def test_a_cached_hit_never_calls_the_detector(tmp_path):
    detector = StubDetector()
    result = query_room(room_on_disk(tmp_path), "sofa", detector=detector, verbose=False)
    assert detector.calls == []
    assert result.source == "cache"


def test_a_cached_miss_calls_the_detector(tmp_path):
    detector = StubDetector()
    result = query_room(
        room_on_disk(tmp_path), "aquarium", detector=detector, verbose=False
    )
    assert detector.calls
    assert result.source == "vlm"


def test_a_qualified_phrase_goes_to_the_detector_even_though_its_noun_is_cached(tmp_path):
    """"sofa" is cached; "the sofa by the window" is a spatial relation the
    labels cannot evaluate, so answering it from the cache would answer a
    different question."""
    detector = StubDetector()
    query_room(
        room_on_disk(tmp_path), "the sofa by the window",
        detector=detector, verbose=False,
    )
    assert detector.calls
    assert set(detector.calls) == {"the sofa by the window"}


def test_force_bypasses_a_cached_hit(tmp_path):
    detector = StubDetector()
    result = query_room(
        room_on_disk(tmp_path), "sofa", detector=detector, force=True, verbose=False
    )
    assert detector.calls
    assert result.source == "vlm"


def test_the_detector_is_asked_once_per_labelled_frame(tmp_path):
    detector = StubDetector()
    room = room_on_disk(tmp_path)
    frames = json.loads((room / "observations.json").read_text())["frames_labeled"]
    query_room(room, "aquarium", detector=detector, verbose=False)
    assert len(detector.calls) == len(frames)


def test_vlm_views_produce_a_usable_box(tmp_path):
    detector = StubDetector()
    result = query_room(
        room_on_disk(tmp_path), "sofa", detector=detector, force=True, verbose=False
    )
    assert result.matches
    assert result.matches[0].obb is not None


def test_a_detector_error_on_one_frame_does_not_kill_the_query(tmp_path):
    """Gemini's free tier 429s often enough that all-or-nothing would routinely
    return nothing, which is indistinguishable from the object being absent."""

    class Flaky:
        def __init__(self):
            self.calls = []

        def locate(self, image, phrase):
            self.calls.append(phrase)
            if len(self.calls) == 1:
                raise RuntimeError("429 rate limited")
            det = sofa_detection()
            return [Detection(label=phrase, box_2d=det.box_2d, confidence=0.85)]

    detector = Flaky()
    result = query_room(
        room_on_disk(tmp_path), "aquarium", detector=detector, verbose=False
    )
    assert len(detector.calls) > 1               # it kept going after the failure
    assert result.matches
    assert any("429" in n for n in result.notes)


def test_locate_prompt_names_the_phrase_and_permits_an_empty_answer():
    from room3d.vlm import LOCATE_PROMPT

    prompt = LOCATE_PROMPT.format(phrase="office chair")
    assert "office chair" in prompt
    assert "empty" in prompt.lower() or "none" in prompt.lower()
