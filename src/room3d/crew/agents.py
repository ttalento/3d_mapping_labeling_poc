"""The four roles in the labeling crew.

Each agent owns one tool and one decision. The division is deliberate: the two
middle roles do work an LLM is genuinely good at (naming what it sees, judging
synonymy), while the geometry they trigger is deterministic and tested. An agent
that "reasons about" a centroid is a bug waiting to happen.
"""

from __future__ import annotations

import os

from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import BaseTool

DEFAULT_MODEL = "gemini/gemini-2.5-flash"


def build_llm(model: str = DEFAULT_MODEL, api_key: str | None = None) -> LLM:
    from ..vlm import resolve_api_key

    key = api_key or resolve_api_key()
    if not key:
        raise RuntimeError("No Gemini API key. Put GEMINI_API_KEY=... (or API_KEY=...) in .env.")
    # LiteLLM reads the key from the environment for some providers regardless
    # of what is passed, so make sure the canonical name is populated.
    os.environ.setdefault("GEMINI_API_KEY", key)
    return LLM(model=model, api_key=key, temperature=0.0)


def build_agents(tools: list[BaseTool], llm: LLM) -> dict[str, Agent]:
    by_name = {t.name: t for t in tools}

    curator = Agent(
        role="Frame Curator",
        goal=(
            "Choose the smallest set of frames that still sees the whole room, so no "
            "vision-model call is spent on a view of a corner already covered."
        ),
        backstory=(
            "You plan surveys. You know a walkthrough lingers in some spots and sweeps "
            "past others, so you judge coverage by where the camera stood, not by how "
            "much footage exists."
        ),
        tools=[by_name["select_frames"]],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        # Each agent owns exactly one tool and one job, so a long ReAct loop is
        # pure quota burn. Cap it.
        max_iter=3,
    )

    observer = Agent(
        role="Scene Observer",
        goal=(
            "Identify every distinct physical object visible in the selected frames, "
            "naming each with a short common noun."
        ),
        backstory=(
            "You catalogue room contents. You name what is actually there rather than "
            "what a room like this usually contains, you count repeats as separate "
            "objects, and you leave out walls, floors and ceilings."
        ),
        tools=[by_name["detect_objects"]],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        # Each agent owns exactly one tool and one job, so a long ReAct loop is
        # pure quota burn. Cap it.
        max_iter=3,
    )

    engineer = Agent(
        role="Projection Engineer",
        goal=(
            "Turn each 2D detection into a 3D position using the reconstruction's "
            "per-frame pointmaps, and reject any detection the geometry does not support."
        ),
        backstory=(
            "You work in 3D reconstruction. You trust the tool's arithmetic over your "
            "own and you would rather emit nothing than a confident wrong coordinate, "
            "because a fabricated position is worse than a missing one."
        ),
        tools=[by_name["project_detections"]],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        # Each agent owns exactly one tool and one job, so a long ReAct loop is
        # pure quota burn. Cap it.
        max_iter=3,
    )

    reconciler = Agent(
        role="Label Reconciler",
        goal=(
            "Merge repeated sightings of the same object into one record, keeping "
            "genuinely different objects apart, and settle on one canonical name each."
        ),
        backstory=(
            "You resolve duplicate records. You know 'monitor', 'computer screen' and "
            "'display' are one thing while 'office chair' and 'armchair' are two, and "
            "that two objects sitting in the same place are still two objects."
        ),
        tools=[by_name["cluster_observations"]],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        # Each agent owns exactly one tool and one job, so a long ReAct loop is
        # pure quota burn. Cap it.
        max_iter=3,
    )

    return {
        "curator": curator,
        "observer": observer,
        "engineer": engineer,
        "reconciler": reconciler,
    }


def build_crew(agents: dict, max_frames: int, max_rpm: int = 10) -> Crew:
    select = Task(
        description=(
            f"Select up to {max_frames} frames covering the room. "
            "Use the select_frames tool. Report the chosen indices."
        ),
        expected_output="The list of selected frame indices.",
        agent=agents["curator"],
    )

    detect = Task(
        description=(
            "Run detect_objects over the selected frames. Report how many objects were "
            "found and the full vocabulary of labels observed."
        ),
        expected_output="A count of detections and the list of distinct labels.",
        agent=agents["observer"],
        context=[select],
    )

    project = Task(
        description=(
            "Run project_detections to lift the detections into 3D. Report how many "
            "observations survived and how many were dropped for weak geometry."
        ),
        expected_output="Counts of surviving and dropped observations.",
        agent=agents["engineer"],
        context=[detect],
    )

    cluster = Task(
        description=(
            "Run cluster_observations to merge observations into unique objects. "
            "Return the tool's JSON output verbatim as your final answer — do not "
            "summarise it, reformat it, or add commentary."
        ),
        expected_output="The JSON array of labeled objects, exactly as the tool returned it.",
        agent=agents["reconciler"],
        context=[project],
    )

    return Crew(
        agents=list(agents.values()),
        tasks=[select, detect, project, cluster],
        process=Process.sequential,
        # Free-tier Gemini allows ~15 RPM. Without this the crew burns through
        # the quota on agent reasoning alone and dies mid-run with a 429.
        max_rpm=max_rpm,
        verbose=True,
    )
