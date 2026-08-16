"""CrewAI tool contracts.

The zero-argument check exists because two tools originally declared none.
Gemini's function-calling API returns an empty completion when handed a function
declaration with an empty parameters object, and CrewAI surfaces that as
"Invalid response from LLM call - None or empty" -- thrown before any tool runs,
with nothing in it to suggest the schema is at fault.
"""

import numpy as np
import pytest

from room3d.artifacts import Reconstruction
from room3d.config import LabelConfig
from room3d.crew.session import LabelingSession
from room3d.crew.tools import (
    ClusterObservationsTool,
    DetectObjectsTool,
    ProjectDetectionsTool,
    SelectFramesTool,
    build_tools,
)

N, H, W = 3, 16, 16


@pytest.fixture
def session():
    recon = Reconstruction(
        images=np.zeros((N, H, W, 3), np.uint8),
        pts3d=np.zeros((N, H, W, 3), np.float32),
        conf_mask=np.ones((N, H, W), bool),
        poses=np.tile(np.eye(4, dtype=np.float32), (N, 1, 1)),
        intrinsics=np.tile(np.eye(3, dtype=np.float32), (N, 1, 1)),
        frame_ids=np.arange(N, dtype=np.int32),
    )
    return LabelingSession(recon=recon, config=LabelConfig())


class StubDetector:
    def detect(self, image):
        return []


def all_tools(session):
    return build_tools(session, StubDetector(), llm=None)


def test_every_tool_declares_at_least_one_argument(session):
    for tool in all_tools(session):
        fields = tool.args_schema.model_fields
        assert fields, (
            f"{tool.name} declares no arguments; Gemini returns an empty "
            "completion for a function declaration with empty parameters"
        )


def test_every_tool_argument_is_optional(session):
    """The agent must be able to call any tool with no arguments at all."""
    for tool in all_tools(session):
        for name, field in tool.args_schema.model_fields.items():
            assert not field.is_required(), f"{tool.name}.{name} is required"


def test_every_tool_argument_is_described(session):
    for tool in all_tools(session):
        for name, field in tool.args_schema.model_fields.items():
            assert field.description, f"{tool.name}.{name} has no description"


def test_tool_names_are_unique_and_stable(session):
    names = [t.name for t in all_tools(session)]
    assert names == ["select_frames", "detect_objects", "project_detections",
                     "cluster_observations"]


def test_every_tool_has_a_description(session):
    for tool in all_tools(session):
        assert tool.description and len(tool.description) > 20


# --- the new arguments actually do something ---------------------------------


def test_detect_objects_honours_explicit_frame_indices(session):
    DetectObjectsTool(session, StubDetector())._run(frame_indices=[0, 2])
    assert session.selected_frames == [0, 2]


def test_detect_objects_ignores_out_of_range_indices(session):
    DetectObjectsTool(session, StubDetector())._run(frame_indices=[0, 99, -1])
    assert session.selected_frames == [0]


def test_detect_objects_falls_back_to_coverage_selection(session):
    DetectObjectsTool(session, StubDetector())._run()
    assert session.selected_frames == [0, 1, 2]


def test_select_frames_zero_means_config_default(session):
    SelectFramesTool(session)._run(max_frames=0)
    assert len(session.selected_frames) == N


def test_select_frames_respects_an_explicit_budget(session):
    SelectFramesTool(session)._run(max_frames=2)
    assert len(session.selected_frames) == 2


def test_cluster_returns_json_even_with_nothing_to_do(session):
    import json

    assert json.loads(ClusterObservationsTool(session, None)._run()) == []


def test_project_with_no_detections_is_harmless(session):
    ProjectDetectionsTool(session)._run()
    assert session.observations == []
