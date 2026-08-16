"""Stage 3 entry point: run the labeling pipeline over a reconstruction.

Two execution modes over the *same* tools:

  crew   -- the four CrewAI agents drive the tools and narrate their reasoning
  direct -- the tools are called in order with no agent loop

`direct` is not a toy. Agent loops add failure modes (a model deciding to
re-run a step, or to paraphrase JSON instead of returning it) that have nothing
to do with whether the geometry is right. When a labeled scene looks wrong, the
first question is always "geometry or agent?", and `--direct` answers it in one
run. It also costs a fraction of the free-tier quota.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..artifacts import Reconstruction
from ..config import LabelConfig
from ..vlm import GeminiDetector, load_env, resolve_model
from .session import LabelingSession
from .tools import (
    ClusterObservationsTool,
    DetectObjectsTool,
    ProjectDetectionsTool,
    SelectFramesTool,
    build_tools,
)


def run_labeling(
    recon: Reconstruction,
    out_path: str | Path,
    *,
    config: LabelConfig | None = None,
    use_crew: bool = True,
    room_name: str = "room",
    scale_verified: bool = False,
    verbose: bool = True,
) -> dict:
    """Label a reconstruction and write objects.json. Returns the result dict."""
    config = config or LabelConfig()
    load_env()

    # A MODEL/GEMINI_MODEL in .env overrides the config default, but never an
    # explicit config value the caller set on purpose.
    if config.model == LabelConfig.model:
        config.model = resolve_model(config.model)
    if verbose:
        print(f"[label] model={config.model}")

    session = LabelingSession(recon=recon, config=config)

    # The crew spends requests on agent reasoning as well as detection, so the
    # detector gets a smaller share of the per-minute budget in crew mode.
    detector_rpm = max(1, config.max_rpm // 2) if use_crew else config.max_rpm
    detector = GeminiDetector(
        model=config.model,
        request_masks=config.use_masks,
        max_retries=config.max_retries,
        min_interval_s=60.0 / detector_rpm,
    )

    if use_crew:
        from .agents import build_agents, build_crew, build_llm

        llm = build_llm(f"gemini/{config.model}")
        tools = build_tools(session, detector, llm)
        crew = build_crew(
            build_agents(tools, llm), config.max_frames, max_rpm=config.max_rpm
        )
        try:
            crew.kickoff()
        except ValueError as exc:
            # CrewAI turns an underlying 429 into "Invalid response from LLM
            # call - None or empty", which says nothing about the real cause and
            # sends you hunting through tool schemas. Name it.
            if "None or empty" in str(exc):
                raise RuntimeError(
                    "The crew's LLM returned nothing. On Gemini's free tier this is "
                    "almost always the per-minute quota (~15 requests/minute): the "
                    "agents' own reasoning calls share it with detection.\n"
                    f"Currently label.max_rpm={config.max_rpm}. Either lower it, wait a "
                    "minute, or run the same pipeline without the agent loop:\n"
                    "    room3d label <frames.npz> --room <name> --direct"
                ) from exc
            raise
    else:
        llm = _try_build_llm(config, verbose)
        SelectFramesTool(session)._run(config.max_frames)
        DetectObjectsTool(session, detector)._run()
        ProjectDetectionsTool(session)._run()
        ClusterObservationsTool(session, llm)._run(use_llm_synonyms=llm is not None)

    result = {
        "room": room_name,
        "units": "meters",
        "scale_verified": scale_verified,
        "n_frames_labeled": len(session.selected_frames),
        "objects": [o.as_dict() for o in session.objects],
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    # The observations are the expensive part of the run -- they cost one VLM
    # call per frame. Persisting them means re-clustering with different
    # thresholds is free, and a viewer can draw an object back onto the exact
    # 2D box it came from.
    obs_path = out_path.parent / "observations.json"
    obs_path.write_text(json.dumps(build_observations_doc(session, room_name), indent=2))

    if verbose:
        print(f"\n[label] {len(session.objects)} objects -> {out_path}")
        print(f"[label] {len(session.observations)} observations -> {obs_path}")

    return result


def build_observations_doc(session: LabelingSession, room_name: str) -> dict:
    """Serialise observations, tagged with the object each was clustered into."""
    owner: dict[int, str] = {}
    for obj in session.objects:
        for oid in obj.observation_ids:
            owner[oid] = obj.id

    return {
        "room": room_name,
        "image_hw": list(session.recon.image_hw),
        "frames_labeled": sorted(session.selected_frames),
        "observations": [
            {"id": i, "object_id": owner.get(i), **obs.as_dict()}
            for i, obs in enumerate(session.observations)
        ],
    }


def _try_build_llm(config: LabelConfig, verbose: bool):
    """The synonym grouping is a nicety; its absence must not fail the run."""
    try:
        from .agents import build_llm

        return build_llm(f"gemini/{config.model}")
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"  [label] no LLM for synonym grouping ({exc}); using static table")
        return None
