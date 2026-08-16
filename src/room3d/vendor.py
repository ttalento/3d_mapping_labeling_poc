"""Put the vendored NAVER repos on sys.path.

`mast3r`, `dust3r` and `croco` ship no pyproject/setup.py, so they cannot be
installed as packages. They live under third_party/ as plain checkouts and are
imported by prepending their roots to sys.path.

Only mast3r is cloned: it bundles dust3r as a submodule, which in turn bundles
croco. Cloning dust3r separately would put two copies of the `dust3r` package on
the path.

Import order matters. croco exposes its package as the very generic name
`models`, so it goes last.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MAST3R = _REPO_ROOT / "third_party" / "mast3r"
_DUST3R = _MAST3R / "dust3r"
_CROCO = _DUST3R / "croco"

_PATHS = (_MAST3R, _DUST3R, _CROCO)

_installed = False


def ensure() -> None:
    """Idempotently prepend the vendored roots to sys.path."""
    global _installed
    if _installed:
        return

    missing = [p for p in _PATHS if not p.is_dir()]
    if missing:
        raise RuntimeError(
            "Vendored dependencies are missing: "
            + ", ".join(str(p) for p in missing)
            + "\nRun: git clone --recursive --depth 1 "
            "https://github.com/naver/mast3r.git third_party/mast3r"
        )

    for p in _PATHS:
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)

    _installed = True


def repo_root() -> Path:
    return _REPO_ROOT
