"""LLM adjudication of label synonymy -- the one judgement call in fusion.

Geometry decides what could be the same object. This decides whether "monitor"
and "computer screen" name the same kind of thing.

The obvious implementation asks the model per candidate pair, which is O(n^2)
calls over the observed vocabulary and slow enough to matter on a free tier.
Instead we ask once: here is every label the VLM produced in this room, group
the ones that mean the same thing. That collapses the whole question into a
single call and a dict lookup, and it gives the model more context to judge
with than an isolated pair ever would.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence

from ..fusion import default_canonicalize, default_label_compatible, normalize_label

GROUPING_PROMPT = """\
These labels were produced by a vision model describing objects in one room.
Group the labels that refer to the same KIND of object.

Labels:
{labels}

Rules:
- Group only true synonyms for the same kind of thing: "monitor" / "computer
  screen" / "display" belong together.
- Do NOT group things that merely sit near each other or are parts of a whole:
  "desk" and "monitor" are separate; "chair" and "cushion" are separate.
- Do NOT group different kinds within a category: "office chair" and "armchair"
  stay separate.
- Every label must appear in exactly one group. Singletons are expected and fine.
- `canonical` must be the most natural short name, and must be one of the labels
  in that group.

Return JSON only, in this exact shape:
{{"groups": [{{"canonical": "monitor", "labels": ["monitor", "display"]}}]}}"""


class SynonymResolver:
    """Maps each observed label to a canonical name, backed by one LLM call."""

    def __init__(self, groups: Sequence[Sequence[str]] | None = None):
        self._canonical: dict[str, str] = {}
        for group in groups or []:
            if not group:
                continue
            canonical = normalize_label(group[0])
            for label in group:
                self._canonical[normalize_label(label)] = canonical

    @property
    def groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for label, canonical in self._canonical.items():
            out.setdefault(canonical, []).append(label)
        return out

    def compatible(self, a: str, b: str) -> bool:
        na, nb = normalize_label(a), normalize_label(b)
        if na == nb:
            return True
        ca, cb = self._canonical.get(na), self._canonical.get(nb)
        if ca is not None and cb is not None:
            return ca == cb
        # Unknown to the LLM grouping: fall back to the static table rather than
        # guessing, so an incomplete grouping cannot silently merge everything.
        return default_label_compatible(a, b)

    def canonicalize(self, labels: Sequence[str]) -> str:
        mapped = [self._canonical.get(normalize_label(l), normalize_label(l)) for l in labels]
        return default_canonicalize(mapped)


def build_synonym_resolver(
    labels: Sequence[str],
    call_llm: Callable[[str], str] | None,
    *,
    verbose: bool = True,
) -> SynonymResolver:
    """One LLM call over the whole observed vocabulary.

    Falls back to the deterministic synonym table if no LLM is supplied or the
    call fails -- a labeling run should degrade, not die, when the model is
    unavailable.
    """
    vocabulary = sorted({normalize_label(l) for l in labels if l and l.strip()})
    if not vocabulary or call_llm is None:
        return SynonymResolver()

    prompt = GROUPING_PROMPT.format(labels="\n".join(f"- {l}" for l in vocabulary))

    try:
        raw = call_llm(prompt)
        groups = _parse_groups(raw, vocabulary)
    except Exception as exc:  # noqa: BLE001 - degrade to the static table
        if verbose:
            print(f"  [semantics] LLM grouping unavailable ({exc}); using static synonyms")
        return SynonymResolver()

    if verbose:
        merged = {c: g for c, g in groups.items() if len(g) > 1}
        print(f"  [semantics] {len(vocabulary)} labels -> {len(groups)} groups"
              + (f"; merged: {merged}" if merged else "; no synonyms found"))

    return SynonymResolver([[canonical, *others] for canonical, others in groups.items()])


def _parse_groups(raw: str, vocabulary: Sequence[str]) -> dict[str, list[str]]:
    """Parse the model's grouping, keeping only labels we actually observed."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text[4:] if text.lower().startswith("json") else text

    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in response")

    data = json.loads(text[start : end + 1])
    known = set(vocabulary)
    seen: set[str] = set()
    groups: dict[str, list[str]] = {}

    for group in data.get("groups", []):
        members = [
            normalize_label(l)
            for l in group.get("labels", [])
            if normalize_label(l) in known
        ]
        members = [m for m in dict.fromkeys(members) if m not in seen]
        if not members:
            continue

        canonical = normalize_label(group.get("canonical", ""))
        if canonical not in members:
            canonical = min(members, key=lambda m: (len(m), m))

        groups[canonical] = [m for m in members if m != canonical]
        seen.update(members)

    # Anything the model dropped stays a singleton rather than vanishing.
    for label in vocabulary:
        if label not in seen:
            groups.setdefault(label, [])

    return groups
