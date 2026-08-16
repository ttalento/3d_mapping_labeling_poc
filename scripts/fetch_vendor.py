"""One-time download of the frontend's only third-party asset.

Vendoring rather than hot-linking a CDN keeps the viewer working offline and
keeps the app reproducible. Only three.js core is needed -- the orbit controls
are ~70 lines in app.js, which is cheaper than vendoring the addons tree and
wiring up an import map for it.

    uv run python scripts/fetch_vendor.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

THREE_VERSION = "0.160.0"
ASSETS = {
    "three.module.js": f"https://unpkg.com/three@{THREE_VERSION}/build/three.module.js",
}

VENDOR = Path(__file__).resolve().parents[1] / "src" / "room3d" / "webapp" / "static" / "vendor"


def main() -> int:
    VENDOR.mkdir(parents=True, exist_ok=True)

    for name, url in ASSETS.items():
        dest = VENDOR / name
        if dest.exists() and dest.stat().st_size > 10_000:
            print(f"  {name}: already present ({dest.stat().st_size / 1024:.0f} KB)")
            continue

        print(f"  {name}: downloading from {url}")
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
        except Exception as exc:  # noqa: BLE001
            print(f"    FAILED: {exc}", file=sys.stderr)
            return 1

        dest.write_bytes(data)
        print(f"    saved {len(data) / 1024:.0f} KB -> {dest}")

    print("\nVendor assets ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
