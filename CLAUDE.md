# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

The repository is currently empty — no source, no build system, no tests. Nothing below describes application architecture because none exists yet. Once code lands, re-run `/init` so this file documents the real build, test, and architecture.

## Model policy for agent dispatch

Set deliberately by the repo owner. It splits thinking work from typing work.

| Work | Model |
|------|-------|
| Main thread (answering the user directly) | Whatever model the user has selected — do not override |
| Planning, architecture, design, research | Opus 5 |
| Implementation, mechanical edits, verification | Sonnet 5 |

How to apply it when spawning agents:

- **`implementer` and `verifier`** (defined in [.claude/agents/](.claude/agents/)) are pinned to Sonnet 5 in their frontmatter. Dispatching them needs no `model` argument.
- **`Plan`** and other design/research dispatches: pass `model: "opus"`, or omit `model` while the main thread is already on Opus.
- **`Explore`, `general-purpose`, `claude`** used for implementation or mechanical work: pass `model: "sonnet"` explicitly. These are built-ins that inherit the parent model otherwise, and the parent is Opus.

The `model` argument to the `Agent` tool overrides an agent definition's frontmatter, so it is the lever for built-ins whose prompts should not be replaced.

The user's `~/.claude/settings.json` sets `"model": "opus"` and `"effortLevel": "high"` globally. That governs the main thread only — Claude Code has no settings key for a default subagent model, which is why the policy lives here and in agent frontmatter rather than in `settings.json`.
